"""
FastAPI 网页应用：用于多人上传图片、识别、反馈和触发重训。
"""

import json
import sys
import threading
import zipfile
import shutil
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINT_PATH, EXCEL_PATH, EXCLUDED_STYLE_CATEGORIES, PROCESSED_DIR, TRAIN_IMAGE_ROOT
from data_processor import DataProcessor
from feedback_loop import (
    append_feedback_log,
    load_recent_feedback,
    load_recent_wrong_feedback,
    save_feedback_image,
)
from inference import StylePredictor
from train import Trainer


BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / 'web_templates'
STATIC_DIR = BASE_DIR / 'web_static'
UPLOAD_DIR = Path(PROCESSED_DIR) / 'web_uploads'
TRAINING_STATUS_PATH = Path(PROCESSED_DIR) / 'training_status.json'
DATASET_IMAGE_DIR = Path(TRAIN_IMAGE_ROOT)
ASSET_UPLOAD_DIR = Path(PROCESSED_DIR) / 'asset_uploads'

STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ASSET_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='室内风格协作标注平台')
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
app.mount('/uploads', StaticFiles(directory=str(UPLOAD_DIR)), name='uploads')
app.mount('/dataset-images', StaticFiles(directory=str(DATASET_IMAGE_DIR)), name='dataset-images')
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_predictor_cache = None
_predictor_init_error = ''
_training_thread = None
_training_lock = threading.Lock()


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


def list_available_styles():
    """合并当前模型风格与本地图库风格，供网页下拉选择。"""
    predictor = get_predictor_if_available()
    style_set = set(predictor.style_to_label.keys()) if predictor is not None else set()

    image_root = Path(TRAIN_IMAGE_ROOT)
    if image_root.exists():
        for path in image_root.iterdir():
            if path.is_dir() and path.name.strip() and path.name not in EXCLUDED_STYLE_CATEGORIES:
                style_set.add(path.name.strip())

    return sorted(style_set)


def write_training_status(status, detail=''):
    payload = {
        'status': status,
        'detail': detail,
        'updated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
    }
    TRAINING_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_training_status():
    if TRAINING_STATUS_PATH.exists():
        return json.loads(TRAINING_STATUS_PATH.read_text(encoding='utf-8'))
    return {
        'status': 'idle',
        'detail': '尚未记录训练任务。',
        'updated_at': '',
    }


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
        'feedback_total': len(load_recent_feedback(limit=100000)),
        'feedback_wrong_total': len(load_recent_wrong_feedback(limit=100000)),
    }

    if processed_path.exists():
        import pandas as pd
        df = pd.read_csv(processed_path, encoding='utf-8')
        snapshot['processed_rows'] = int(len(df))
        snapshot['processed_styles'] = int(df['style'].nunique()) if 'style' in df.columns else 0

    if snapshot['model_updated_at'] is not None:
        from datetime import datetime
        snapshot['model_updated_at_text'] = datetime.fromtimestamp(snapshot['model_updated_at']).strftime('%Y-%m-%d %H:%M:%S')
    else:
        snapshot['model_updated_at_text'] = '未生成'

    return snapshot


def build_context(**extra):
    predictor = get_predictor_if_available()
    recent_wrong_feedback = []
    for row in load_recent_wrong_feedback(limit=12):
        row = dict(row)
        row['saved_url'] = dataset_image_url(row.get('saved_path', ''))
        recent_wrong_feedback.append(row)

    base = {
        'styles': list_available_styles(),
        'recent_feedback': load_recent_feedback(limit=15),
        'recent_wrong_feedback': recent_wrong_feedback,
        'training_snapshot': build_training_snapshot(),
        'predictor_is_stale': predictor.is_stale if predictor is not None else False,
        'predictor_stale_reason': predictor.stale_reason if predictor is not None else '',
        'predictor_available': predictor is not None,
        'predictor_error': _predictor_init_error,
        'message': '',
        'error': '',
        'result': None,
        'asset_uploads': {
            'processed_bundle': (ASSET_UPLOAD_DIR / 'render_processed_bundle.zip').exists(),
            'model_bundle': (ASSET_UPLOAD_DIR / 'render_model_bundle.zip').exists(),
            'excel_bundle': (ASSET_UPLOAD_DIR / 'render_excel_bundle.zip').exists(),
        },
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


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(),
    )


@app.head('/')
async def index_head():
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
    saved_path = save_feedback_image(
        predictor=predictor,
        image_input=str(image_path),
        style_name=final_style,
        image=image,
    )
    append_feedback_log(
        image_input=str(image_path),
        predictions=predictions,
        chosen_style=final_style,
        saved_path=saved_path,
        is_correct=is_correct,
    )

    message = f'反馈已记录，图片已归档到 {saved_path}'
    if retrain_now == 'y':
        started = start_background_incremental_training()
        if started:
            message += '；后台快速微调已经启动，你可以去训练状态页查看进度。'
        else:
            message += '；当前已有训练任务在后台运行，这次没有重复启动。'
    else:
        write_training_status('idle', '最近一次反馈已记录，尚未触发重训。')

    if image_path.exists():
        image_path.unlink()

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context=build_context(message=message),
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
        label_map = {
            'processed': 'processed zip',
            'model': 'model zip',
            'excel': 'excel zip',
        }
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(message=f'{label_map[bundle_kind]} 已上传到云端暂存区。'),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(error=f'资产上传失败：{exc}'),
        )


@app.post('/upload-assets', response_class=HTMLResponse)
async def upload_assets(request: Request):
    try:
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
        _copy_tree_contents(processed_src, Path(PROCESSED_DIR))
        Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, Path(CHECKPOINT_PATH))
        shutil.copy2(excel_src, Path(EXCEL_PATH))

        refresh_predictor()
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context=build_context(message='云端资产已上传完成，模型已重新加载，识别功能已恢复。'),
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
        context=build_context(),
    )


@app.head('/training-status')
async def training_status_head():
    return Response(status_code=200)


@app.get('/healthz')
async def healthz():
    predictor = get_predictor_if_available()
    return {
        'ok': True,
        'predictor_ready': predictor is not None,
        'num_styles': predictor.num_classes if predictor is not None else 0,
        'predictor_error': _predictor_init_error,
    }
