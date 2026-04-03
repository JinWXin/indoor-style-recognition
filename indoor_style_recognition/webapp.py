"""
FastAPI 网页应用：用于多人上传图片、识别、反馈和触发重训。
"""

import hmac
import json
import os
import sys
import threading
import zipfile
import shutil
from typing import Iterable
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ADMIN_PASSWORD,
    CHECKPOINT_PATH,
    EXCEL_PATH,
    EXCLUDED_STYLE_CATEGORIES,
    PROCESSED_DIR,
    SESSION_SECRET,
    STYLE_DEFINITION_SHEET_NAME,
    TRAIN_IMAGE_ROOT,
)
from data_processor import DataProcessor
from feedback_loop import (
    append_feedback_log,
    append_pending_review_log,
    load_recent_feedback,
    load_pending_reviews,
    load_recent_wrong_feedback,
    save_feedback_image,
    save_pending_review_image,
    update_pending_review_status,
)
from inference import StylePredictor
from style_knowledge import STYLE_TEXT_COLUMNS, normalize_text
from train import Trainer


BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / 'web_templates'
STATIC_DIR = BASE_DIR / 'web_static'
UPLOAD_DIR = Path(PROCESSED_DIR) / 'web_uploads'
TRAINING_STATUS_PATH = Path(PROCESSED_DIR) / 'training_status.json'
DATASET_IMAGE_DIR = Path(TRAIN_IMAGE_ROOT)
ASSET_UPLOAD_DIR = Path(PROCESSED_DIR) / 'asset_uploads'
ASSET_RESTORE_STATUS_PATH = Path(PROCESSED_DIR) / 'asset_restore_status.json'
REVIEW_QUEUE_DIR = Path(PROCESSED_DIR) / 'review_queue'
REVIEW_IMAGE_DIR = REVIEW_QUEUE_DIR / 'images'

STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ASSET_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='室内风格协作标注平台')
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie='style_admin_session',
    same_site='lax',
    https_only=os.getenv('STYLE_SESSION_HTTPS_ONLY', '0') == '1',
)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
app.mount('/uploads', StaticFiles(directory=str(UPLOAD_DIR)), name='uploads')
app.mount('/dataset-images', StaticFiles(directory=str(DATASET_IMAGE_DIR)), name='dataset-images')
app.mount('/review-images', StaticFiles(directory=str(REVIEW_IMAGE_DIR)), name='review-images')
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_predictor_cache = None
_predictor_init_error = ''
_training_thread = None
_training_lock = threading.Lock()
_asset_restore_thread = None
_asset_restore_lock = threading.Lock()
_style_definition_cache = {
    'excel_path': '',
    'mtime': None,
    'profiles': {},
}


def _safe_error_message(prefix, exc):
    return f'{prefix}: {exc}'


def is_admin_authenticated(request: Request):
    return bool(request.session.get('is_admin_authenticated'))


def verify_admin_password(password: str):
    return hmac.compare_digest(password or '', ADMIN_PASSWORD)


def require_admin(request: Request):
    if is_admin_authenticated(request):
        return None
    return RedirectResponse(url='/admin/login', status_code=303)


def get_predictor():
    global _predictor_cache, _predictor_init_error
    if _predictor_cache is None:
        try:
            _predictor_cache = StylePredictor(validate_freshness=False)
            _predictor_init_error = ''
        except Exception as exc:
            _predictor_init_error = str(exc)
            raise
    return _predictor_cache


def refresh_predictor():
    global _predictor_cache, _predictor_init_error
    _predictor_cache = StylePredictor(validate_freshness=False)
    _predictor_init_error = ''
    return _predictor_cache


def get_predictor_if_available():
    try:
        return get_predictor()
    except Exception:
        return None


def predictor_artifacts_ready():
    required_paths = [
        Path(PROCESSED_DIR) / 'style_mapping.json',
        Path(PROCESSED_DIR) / 'style_profiles.json',
        Path(PROCESSED_DIR) / 'style_text_features.npz',
        Path(CHECKPOINT_PATH),
    ]
    return all(path.exists() for path in required_paths)


def dataset_image_url(path_value):
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(Path(TRAIN_IMAGE_ROOT).resolve())
        return f"/dataset-images/{rel.as_posix()}"
    except Exception:
        return ''


def upload_image_url(path_value):
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(UPLOAD_DIR.resolve())
        return f"/uploads/{rel.as_posix()}"
    except Exception:
        return ''


def review_image_url(path_value):
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(REVIEW_IMAGE_DIR.resolve())
        return f"/review-images/{rel.as_posix()}"
    except Exception:
        return ''


def list_available_styles():
    """合并当前模型风格与本地图库风格，供网页下拉选择。"""
    predictor = _predictor_cache
    style_set = set(predictor.style_to_label.keys()) if predictor is not None else set()

    image_root = Path(TRAIN_IMAGE_ROOT)
    if image_root.exists():
        for path in image_root.iterdir():
            if path.is_dir() and path.name.strip() and path.name not in EXCLUDED_STYLE_CATEGORIES:
                style_set.add(path.name.strip())

    return sorted(style_set)


def _human_size(num_bytes):
    value = float(num_bytes or 0)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024 or unit == 'GB':
            return f"{value:.0f} {unit}" if unit == 'B' else f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(num_bytes)} B"


def load_style_definition_profiles():
    excel_path = Path(EXCEL_PATH)
    if not excel_path.exists():
        return {}

    mtime = excel_path.stat().st_mtime
    if (
        _style_definition_cache['excel_path'] == str(excel_path) and
        _style_definition_cache['mtime'] == mtime
    ):
        return _style_definition_cache['profiles']

    definition_df = pd.read_excel(excel_path, sheet_name=STYLE_DEFINITION_SHEET_NAME)
    if '美学风格词' not in definition_df.columns:
        return {}

    indexed = definition_df.copy()
    indexed['美学风格词'] = indexed['美学风格词'].apply(normalize_text)
    indexed = indexed[indexed['美学风格词'] != ''].drop_duplicates(subset=['美学风格词'], keep='first')

    profiles = {}
    for _, row in indexed.iterrows():
        style_name = row.get('美学风格词', '')
        if not style_name:
            continue
        profile = {'style': style_name}
        for column, alias in STYLE_TEXT_COLUMNS:
            profile[alias] = normalize_text(row[column]) if column in row.index else ''
        profiles[style_name] = profile

    _style_definition_cache['excel_path'] = str(excel_path)
    _style_definition_cache['mtime'] = mtime
    _style_definition_cache['profiles'] = profiles
    return profiles


def get_style_profile(style_name):
    if not style_name:
        return {}

    profile = load_style_definition_profiles().get(style_name)
    if profile:
        return profile

    return {
        'style': style_name,
        '定位': '',
        '定位依据': '',
        '定义': '',
        '色彩': '',
        '材料': '',
        '形态': '',
        '风格区别': '',
        '典型特征': '',
        '情绪氛围': '',
        '搜索关键词': '',
    }


def list_style_gallery_items(style_name):
    style_dir = Path(TRAIN_IMAGE_ROOT) / style_name
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    root = Path(TRAIN_IMAGE_ROOT).resolve()
    items = []

    if not style_dir.exists():
        return items

    for image_path in sorted(style_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue
        rel = image_path.resolve().relative_to(root)
        stat = image_path.stat()
        items.append({
            'name': image_path.name,
            'relative_path': rel.as_posix(),
            'url': f"/dataset-images/{rel.as_posix()}",
            'size_text': _human_size(stat.st_size),
            'updated_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        })

    return items


def build_style_sidebar(search_query='', current_style=''):
    query = (search_query or '').strip().lower()
    styles = []
    for style_name in list_available_styles():
        image_count = len(list_style_gallery_items(style_name))
        if query and query not in style_name.lower():
            continue
        styles.append({
            'name': style_name,
            'image_count': image_count,
            'active': style_name == current_style,
        })
    return styles


def resolve_dataset_relative_path(relative_path):
    root = Path(TRAIN_IMAGE_ROOT).resolve()
    candidate = (root / relative_path).resolve()
    candidate.relative_to(root)
    return candidate


def unique_destination_path(directory: Path, filename: str):
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem or 'image'
    suffix = Path(filename).suffix or '.jpg'
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def build_style_gallery_context(request: Request, style_name='', search_query='', message='', error=''):
    sidebar = build_style_sidebar(search_query=search_query, current_style=style_name)
    selected_style = style_name or (sidebar[0]['name'] if sidebar else '')
    if selected_style and not any(item['name'] == selected_style for item in sidebar):
        sidebar = build_style_sidebar(search_query='', current_style=selected_style)

    selected_profile = get_style_profile(selected_style) if selected_style else {}
    selected_images = list_style_gallery_items(selected_style) if selected_style else []

    return build_context(
        skip_predictor_probe=True,
        admin_logged_in=is_admin_authenticated(request),
        message=message,
        error=error,
        style_search_query=search_query,
        style_sidebar=sidebar,
        selected_style=selected_style,
        selected_style_profile=selected_profile,
        selected_style_images=selected_images,
        selected_style_image_count=len(selected_images),
        style_profile_fields=[alias for _, alias in STYLE_TEXT_COLUMNS if alias != '搜索关键词'],
    )


def write_training_status(status, detail=''):
    payload = {
        'status': status,
        'detail': detail,
        'updated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
    }
    TRAINING_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_asset_restore_status(status, detail=''):
    payload = {
        'status': status,
        'detail': detail,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    ASSET_RESTORE_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_asset_restore_status():
    if ASSET_RESTORE_STATUS_PATH.exists():
        return json.loads(ASSET_RESTORE_STATUS_PATH.read_text(encoding='utf-8'))
    return {
        'status': 'idle',
        'detail': '尚未开始上传云端资产。',
        'updated_at': '',
    }


def load_training_status():
    if TRAINING_STATUS_PATH.exists():
        return json.loads(TRAINING_STATUS_PATH.read_text(encoding='utf-8'))
    return {
        'status': 'idle',
        'detail': '尚未记录训练任务。',
        'updated_at': '',
    }


def load_recent_feedback_safe(limit=20):
    try:
        return load_recent_feedback(limit=limit)
    except Exception:
        return []


def load_recent_wrong_feedback_safe(limit=20):
    try:
        return load_recent_wrong_feedback(limit=limit)
    except Exception:
        return []


def load_pending_reviews_safe(limit=50):
    try:
        return load_pending_reviews(limit=limit, status='pending')
    except Exception:
        return []


def run_incremental_training_workflow():
    """后台执行数据处理与快速微调。"""
    try:
        write_training_status('running', '正在重建数据集...')
        processor = DataProcessor()
        processor.process_data()

        write_training_status('running', '正在基于最近反馈做快速微调...')
        trainer = Trainer(num_workers_override=0)
        trainer.run_incremental()

        write_training_status('running', '正在重新加载最新模型...')
        refresh_predictor()
        write_training_status('completed', '后台快速微调已完成，最新模型已加载。')
    except Exception as e:
        write_training_status('failed', f'后台训练失败: {e}')


def start_background_incremental_training():
    """如果当前没有训练任务，则启动一个后台训练线程。"""
    global _training_thread
    with _training_lock:
        if _training_thread is not None and _training_thread.is_alive():
            return False

        _training_thread = threading.Thread(
            target=run_incremental_training_workflow,
            name='style-incremental-training',
            daemon=True,
        )
        _training_thread.start()
        return True


def build_training_snapshot():
    processed_path = Path(PROCESSED_DIR) / 'processed_data.csv'
    mapping_path = Path(PROCESSED_DIR) / 'style_mapping.json'
    model_path = Path(CHECKPOINT_PATH)

    snapshot = {
        'processed_exists': processed_path.exists(),
        'mapping_exists': mapping_path.exists(),
        'model_exists': model_path.exists(),
        'processed_rows': 0,
        'processed_styles': 0,
        'model_updated_at': model_path.stat().st_mtime if model_path.exists() else None,
        'training_status': load_training_status(),
        'feedback_total': len(load_recent_feedback_safe(limit=100000)),
        'feedback_wrong_total': len(load_recent_wrong_feedback_safe(limit=100000)),
        'snapshot_error': '',
    }

    if processed_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(processed_path, encoding='utf-8')
            snapshot['processed_rows'] = int(len(df))
            snapshot['processed_styles'] = int(df['style'].nunique()) if 'style' in df.columns else 0
        except Exception as exc:
            snapshot['snapshot_error'] = _safe_error_message('processed 数据读取失败', exc)

    if snapshot['model_updated_at'] is not None:
        from datetime import datetime
        snapshot['model_updated_at_text'] = datetime.fromtimestamp(snapshot['model_updated_at']).strftime('%Y-%m-%d %H:%M:%S')
    else:
        snapshot['model_updated_at_text'] = '未生成'

    return snapshot


def build_context(skip_predictor_probe=False, **extra):
    asset_restore_status = load_asset_restore_status()
    should_probe_predictor = (
        not skip_predictor_probe and
        asset_restore_status.get('status') != 'running'
    )
    predictor = get_predictor_if_available() if should_probe_predictor else _predictor_cache
    recent_wrong_feedback = []
    for row in load_recent_wrong_feedback_safe(limit=12):
        row = dict(row)
        row['saved_url'] = dataset_image_url(row.get('saved_path', ''))
        recent_wrong_feedback.append(row)

    pending_reviews = []
    for row in load_pending_reviews_safe(limit=20):
        row = dict(row)
        row['pending_url'] = review_image_url(row.get('pending_path', ''))
        pending_reviews.append(row)

    base = {
        'styles': list_available_styles(),
        'recent_feedback': load_recent_feedback_safe(limit=15),
        'recent_wrong_feedback': recent_wrong_feedback,
        'pending_reviews': pending_reviews,
        'training_snapshot': build_training_snapshot(),
        'predictor_is_stale': predictor.is_stale if predictor is not None else False,
        'predictor_stale_reason': predictor.stale_reason if predictor is not None else '',
        'predictor_available': predictor is not None,
        'predictor_error': _predictor_init_error,
        'asset_restore_status': asset_restore_status,
        'message': '',
        'error': '',
        'result': None,
        'asset_uploads': {
            'processed_bundle': (ASSET_UPLOAD_DIR / 'render_processed_bundle.zip').exists(),
            'model_bundle': (ASSET_UPLOAD_DIR / 'render_model_bundle.zip').exists(),
            'excel_bundle': (ASSET_UPLOAD_DIR / 'render_excel_bundle.zip').exists(),
        },
        'admin_logged_in': False,
    }
    base.update(extra)
    return base


def _extract_zip_bytes(binary: bytes, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(binary)) as archive:
        archive.extractall(destination)


def _copy_tree_contents(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _extract_zip_file(zip_path: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def _find_single_excel_file(root: Path) -> Path:
    excel_files = sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.xlsx', '.xls'}
    )
    if not excel_files:
        raise FileNotFoundError('excel zip 中缺少 xlsx/xls 文件')
    return excel_files[0]


def _save_uploaded_file(upload: UploadFile, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with destination.open('wb') as handle:
        shutil.copyfileobj(upload.file, handle)


def _apply_uploaded_assets():
    processed_zip = ASSET_UPLOAD_DIR / 'render_processed_bundle.zip'
    model_zip = ASSET_UPLOAD_DIR / 'render_model_bundle.zip'
    excel_zip = ASSET_UPLOAD_DIR / 'render_excel_bundle.zip'

    missing = [
        label for label, path in (
            ('processed zip', processed_zip),
            ('model zip', model_zip),
            ('excel zip', excel_zip),
        ) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"还缺这些文件：{', '.join(missing)}")

    tmp_root = Path(PROCESSED_DIR) / 'asset_imports'
    tmp_root.mkdir(parents=True, exist_ok=True)
    processed_unpack = tmp_root / 'processed_unpack'
    model_unpack = tmp_root / 'model_unpack'
    excel_unpack = tmp_root / 'excel_unpack'

    write_asset_restore_status('running', '正在解压 processed / model / excel 三个资产包...')
    _extract_zip_file(processed_zip, processed_unpack)
    _extract_zip_file(model_zip, model_unpack)
    _extract_zip_file(excel_zip, excel_unpack)

    processed_src = processed_unpack / 'indoor_style_recognition' / 'data' / 'processed'
    model_src = model_unpack / 'indoor_style_recognition' / 'models' / 'best_model.pth'
    excel_src = _find_single_excel_file(excel_unpack)

    if not processed_src.exists():
        raise FileNotFoundError('processed zip 中缺少 indoor_style_recognition/data/processed')
    if not model_src.exists():
        raise FileNotFoundError('model zip 中缺少 indoor_style_recognition/models/best_model.pth')

    write_asset_restore_status('running', '正在把处理文件、模型和 Excel 复制到 Render 磁盘...')
    _copy_tree_contents(processed_src, Path(PROCESSED_DIR))
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, Path(CHECKPOINT_PATH))
    shutil.copy2(excel_src, Path(EXCEL_PATH))

    write_asset_restore_status('running', '文件复制完成，正在重新加载模型...')
    refresh_predictor()
    write_asset_restore_status('completed', '云端资产已恢复完成，模型已重新加载。')


def run_background_asset_restore():
    try:
        _apply_uploaded_assets()
    except Exception as exc:
        write_asset_restore_status('failed', f'云端资产恢复失败：{exc}')


def start_background_asset_restore():
    global _asset_restore_thread
    with _asset_restore_lock:
        if _asset_restore_thread is not None and _asset_restore_thread.is_alive():
            return False

        _asset_restore_thread = threading.Thread(
            target=run_background_asset_restore,
            name='asset-restore-worker',
            daemon=True,
        )
        _asset_restore_thread.start()
        return True


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(skip_predictor_probe=True),
    )


@app.head('/')
async def index_head():
    return Response(status_code=200)


@app.get('/styles', response_class=HTMLResponse)
async def style_gallery(
    request: Request,
    style: str = '',
    q: str = '',
):
    return templates.TemplateResponse(
        request=request,
        name='style_gallery.html',
        context=build_style_gallery_context(
            request=request,
            style_name=style.strip(),
            search_query=q.strip(),
        ),
    )


@app.post('/styles/upload', response_class=HTMLResponse)
async def upload_style_images(
    request: Request,
    style_name: str = Form(...),
    search_query: str = Form(''),
    image_files: list[UploadFile] = File(...),
):
    redirect = require_admin(request)
    if redirect is not None:
        return redirect

    saved_count = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    target_dir = Path(TRAIN_IMAGE_ROOT) / style_name.strip()

    try:
        if not style_name.strip():
            raise ValueError('请先选择要上传到哪个风格。')

        for upload in image_files:
            suffix = Path(upload.filename or '').suffix.lower()
            if suffix not in image_extensions:
                continue
            destination = unique_destination_path(target_dir, upload.filename or f'upload{suffix}')
            _save_uploaded_file(upload, destination)
            saved_count += 1

        if saved_count == 0:
            raise ValueError('没有上传成功的图片，请确认文件格式为 jpg/png/webp/bmp。')

        context = build_style_gallery_context(
            request=request,
            style_name=style_name.strip(),
            search_query=search_query.strip(),
            message=f'已上传 {saved_count} 张图片到风格「{style_name.strip()}」。',
        )
    except Exception as exc:
        context = build_style_gallery_context(
            request=request,
            style_name=style_name.strip(),
            search_query=search_query.strip(),
            error=f'上传图片失败：{exc}',
        )

    return templates.TemplateResponse(request=request, name='style_gallery.html', context=context)


@app.post('/styles/delete', response_class=HTMLResponse)
async def delete_style_image(
    request: Request,
    style_name: str = Form(...),
    search_query: str = Form(''),
    relative_path: str = Form(...),
):
    redirect = require_admin(request)
    if redirect is not None:
        return redirect

    try:
        target_path = resolve_dataset_relative_path(relative_path)
        target_path.unlink(missing_ok=True)
        context = build_style_gallery_context(
            request=request,
            style_name=style_name.strip(),
            search_query=search_query.strip(),
            message=f'已删除图片：{Path(relative_path).name}',
        )
    except Exception as exc:
        context = build_style_gallery_context(
            request=request,
            style_name=style_name.strip(),
            search_query=search_query.strip(),
            error=f'删除图片失败：{exc}',
        )

    return templates.TemplateResponse(request=request, name='style_gallery.html', context=context)


@app.post('/styles/move', response_class=HTMLResponse)
async def move_style_image(
    request: Request,
    style_name: str = Form(...),
    target_style: str = Form(...),
    search_query: str = Form(''),
    relative_path: str = Form(...),
):
    redirect = require_admin(request)
    if redirect is not None:
        return redirect

    try:
        if not target_style.strip():
            raise ValueError('请选择目标风格。')

        source_path = resolve_dataset_relative_path(relative_path)
        destination_dir = Path(TRAIN_IMAGE_ROOT) / target_style.strip()
        destination = unique_destination_path(destination_dir, source_path.name)
        shutil.move(str(source_path), str(destination))
        context = build_style_gallery_context(
            request=request,
            style_name=target_style.strip(),
            search_query=search_query.strip(),
            message=f'已把图片移动到风格「{target_style.strip()}」。',
        )
    except Exception as exc:
        context = build_style_gallery_context(
            request=request,
            style_name=style_name.strip(),
            search_query=search_query.strip(),
            error=f'更改风格失败：{exc}',
        )

    return templates.TemplateResponse(request=request, name='style_gallery.html', context=context)


@app.get('/admin/login', response_class=HTMLResponse)
async def admin_login(request: Request):
    if is_admin_authenticated(request):
        return RedirectResponse(url='/admin/reviews', status_code=303)
    return templates.TemplateResponse(
        request=request,
        name='admin_login.html',
        context=build_context(skip_predictor_probe=True),
    )


@app.post('/admin/login', response_class=HTMLResponse)
async def admin_login_submit(
    request: Request,
    password: str = Form(...),
):
    if verify_admin_password(password):
        request.session['is_admin_authenticated'] = True
        return RedirectResponse(url='/admin/reviews', status_code=303)

    return templates.TemplateResponse(
        request=request,
        name='admin_login.html',
        context=build_context(
            skip_predictor_probe=True,
            error='管理员密码不正确，请重试。',
        ),
        status_code=401,
    )


@app.post('/admin/logout')
async def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/admin/login', status_code=303)


@app.get('/admin/reviews', response_class=HTMLResponse)
async def admin_reviews(request: Request):
    redirect = require_admin(request)
    if redirect is not None:
        return redirect
    return templates.TemplateResponse(
        request=request,
        name='admin_reviews.html',
        context=build_context(skip_predictor_probe=True, admin_logged_in=True),
    )


@app.head('/admin/reviews')
async def admin_reviews_head():
    return Response(status_code=200)


@app.post('/predict', response_class=HTMLResponse)
async def predict(
    request: Request,
    image_file: UploadFile = File(...),
):
    predictor = get_predictor_if_available()
    if predictor is None:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(
                error='当前云端模型尚未准备好，网站已经上线，但还缺少 processed 数据或模型文件。'
            ),
        )
    suffix = Path(image_file.filename or 'upload.jpg').suffix or '.jpg'
    upload_path = UPLOAD_DIR / f"upload_{Path(image_file.filename or 'image').stem}{suffix}"
    binary = await image_file.read()
    upload_path.write_bytes(binary)

    image = Image.open(upload_path).convert('RGB')
    result = predictor.predict(image, top_k=3, threshold=0.1)
    result['upload_path'] = str(upload_path.resolve())
    result['upload_name'] = image_file.filename or upload_path.name
    result['upload_url'] = upload_image_url(result['upload_path'])

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(
            result=result,
            message='识别完成，请确认结果或提交正确风格。',
        ),
    )


@app.post('/feedback', response_class=HTMLResponse)
async def feedback(
    request: Request,
    upload_path: str = Form(...),
    predicted_style: str = Form(...),
    pred_1_style: str = Form(''),
    pred_1_confidence: str = Form(''),
    pred_2_style: str = Form(''),
    pred_2_confidence: str = Form(''),
    pred_3_style: str = Form(''),
    pred_3_confidence: str = Form(''),
    feedback_action: str = Form(...),
    chosen_style: str = Form(''),
    new_style_name: str = Form(''),
    retrain_now: str = Form('n'),
):
    predictor = get_predictor_if_available()
    if predictor is None:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(error='当前模型尚未准备好，暂时无法提交识别反馈。'),
        )
    image_path = Path(upload_path)
    if not image_path.exists():
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(error='上传的临时图片不存在，请重新上传。'),
        )

    predictions = [
        {'style': pred_1_style, 'confidence': float(pred_1_confidence or 0.0)},
        {'style': pred_2_style, 'confidence': float(pred_2_confidence or 0.0)},
        {'style': pred_3_style, 'confidence': float(pred_3_confidence or 0.0)},
    ]

    is_correct = feedback_action == 'correct'
    final_style = predicted_style if is_correct else (new_style_name.strip() or chosen_style.strip())
    if not final_style:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(error='请选择正确风格，或者输入一个新的风格名称后再提交反馈。'),
        )

    image = predictor.load_image(str(image_path))
    pending_path = save_pending_review_image(
        predictor=predictor,
        image_input=str(image_path),
        style_name=final_style,
        image=image,
    )
    review_id = append_pending_review_log(
        image_input=str(image_path),
        predictions=predictions,
        chosen_style=final_style,
        pending_path=pending_path,
        is_correct=is_correct,
    )

    message = f'反馈已进入待审核队列，审核编号 {review_id}。审核通过后才会加入训练库。'
    write_training_status('idle', '最近一次反馈已进入待审核队列，尚未触发重训。')

    if image_path.exists():
        image_path.unlink()

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(message=message),
    )


@app.post('/review-feedback', response_class=HTMLResponse)
async def review_feedback(
    request: Request,
    review_id: str = Form(...),
    review_action: str = Form(...),
    review_note: str = Form(''),
):
    redirect = require_admin(request)
    if redirect is not None:
        return redirect
    predictor = get_predictor_if_available()
    if predictor is None:
        return templates.TemplateResponse(
            request=request,
            name='admin_reviews.html',
            context=build_context(error='当前模型尚未准备好，暂时无法处理审核。', admin_logged_in=True),
        )

    pending_rows = load_pending_reviews_safe(limit=500)
    row = next((item for item in pending_rows if item.get('review_id') == review_id), None)
    if row is None:
        return templates.TemplateResponse(
            request=request,
            name='admin_reviews.html',
            context=build_context(error=f'未找到待审核记录：{review_id}', admin_logged_in=True),
        )

    pending_path = Path(row.get('pending_path', ''))
    if not pending_path.exists():
        return templates.TemplateResponse(
            request=request,
            name='admin_reviews.html',
            context=build_context(error=f'待审核图片不存在：{pending_path}', admin_logged_in=True),
        )

    if review_action == 'approve':
        image = predictor.load_image(str(pending_path))
        saved_path = save_feedback_image(
            predictor=predictor,
            image_input=str(pending_path),
            style_name=row.get('chosen_style', '').strip(),
            image=image,
        )
        predictions = []
        for idx in range(1, 4):
            predictions.append({
                'style': row.get(f'pred_{idx}_style', ''),
                'confidence': float(row.get(f'pred_{idx}_confidence', 0.0) or 0.0),
            })
        append_feedback_log(
            image_input=row.get('image_source', ''),
            predictions=predictions,
            chosen_style=row.get('chosen_style', ''),
            saved_path=saved_path,
            is_correct=row.get('is_correct') == '1',
        )
        update_pending_review_status(
            review_id=review_id,
            status='approved',
            review_note=review_note,
            approved_saved_path=saved_path,
        )
        pending_path.unlink(missing_ok=True)
        message = f'审核已通过，图片已加入训练库：{saved_path}'
    elif review_action == 'reject':
        update_pending_review_status(
            review_id=review_id,
            status='rejected',
            review_note=review_note,
            approved_saved_path='',
        )
        pending_path.unlink(missing_ok=True)
        message = f'审核已驳回：{review_id}'
    else:
        return templates.TemplateResponse(
            request=request,
            name='admin_reviews.html',
            context=build_context(error='未知审核动作，请重试。', admin_logged_in=True),
        )

    return templates.TemplateResponse(
        request=request,
        name='admin_reviews.html',
        context=build_context(message=message, admin_logged_in=True),
    )


@app.post('/refresh-model', response_class=HTMLResponse)
async def refresh_model(request: Request):
    try:
        refresh_predictor()
        message = '模型已重新加载。'
        error = ''
    except Exception as exc:
        message = ''
        error = f'模型重新加载失败：{exc}'
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(message=message, error=error),
    )


@app.post('/upload-asset-bundle', response_class=HTMLResponse)
async def upload_asset_bundle(
    request: Request,
    bundle_kind: str = Form(...),
    bundle_file: UploadFile = File(...),
):
    try:
        bundle_map = {
            'processed': 'render_processed_bundle.zip',
            'model': 'render_model_bundle.zip',
            'excel': 'render_excel_bundle.zip',
        }
        if bundle_kind not in bundle_map:
            raise ValueError('未知资产类型，请重新选择。')
        if not (bundle_file.filename or '').lower().endswith('.zip'):
            raise ValueError('请上传 zip 文件。')

        destination = ASSET_UPLOAD_DIR / bundle_map[bundle_kind]
        _save_uploaded_file(bundle_file, destination)
        write_asset_restore_status('idle', f'{bundle_kind} 资产包已上传，等待应用。')
        label_map = {
            'processed': 'processed zip',
            'model': 'model zip',
            'excel': 'excel zip',
        }
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(
                skip_predictor_probe=True,
                message=f'{label_map[bundle_kind]} 已上传到云端暂存区。',
            ),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(
                skip_predictor_probe=True,
                error=f'资产上传失败：{exc}',
            ),
        )


@app.post('/upload-assets', response_class=HTMLResponse)
async def upload_assets(request: Request):
    try:
        started = start_background_asset_restore()
        message = (
            '云端资产恢复任务已经启动，请等待 10-60 秒后刷新页面。'
            if started else
            '云端资产恢复任务已经在后台运行，请稍候刷新页面查看结果。'
        )
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(
                skip_predictor_probe=True,
                message=message,
            ),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(error=f'云端资产上传失败：{exc}'),
        )


@app.get('/training-status', response_class=HTMLResponse)
async def training_status(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='training_status.html',
        context=build_context(skip_predictor_probe=True),
    )


@app.head('/training-status')
async def training_status_head():
    return Response(status_code=200)


@app.get('/healthz')
async def healthz():
    return {
        'ok': True,
        'predictor_ready': _predictor_cache is not None,
        'predictor_files_ready': predictor_artifacts_ready(),
        'num_styles': _predictor_cache.num_classes if _predictor_cache is not None else 0,
        'predictor_error': _predictor_init_error,
    }
