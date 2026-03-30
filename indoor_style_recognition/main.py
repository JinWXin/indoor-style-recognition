"""
主脚本：完整的工作流
"""

import os
import sys

# 添加当前目录到路径
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_processor import DataProcessor
from feedback_loop import append_feedback_log, save_feedback_image
from train import Trainer
from inference import StylePredictor


def prompt_feedback_and_maybe_retrain(predictor, image_input, result):
    """识别后收集人工反馈，并可选择重新训练。"""
    feedback = input("反馈结果是否正确? (y=正确 / n=错误 / skip=跳过): ").strip().lower()
    if feedback in {'', 'skip', 's'}:
        return predictor

    if feedback == 'y':
        chosen_style = result['final_prediction']['style']
        is_correct = True
    elif feedback == 'n':
        print("\n当前可用风格列表:")
        styles = sorted(predictor.style_to_label.keys())
        for idx in range(0, len(styles), 6):
            print("  " + " | ".join(styles[idx:idx + 6]))
        while True:
            chosen_style = input("\n请输入正确风格名称: ").strip()
            if chosen_style:
                break
            print("✗ 风格名称不能为空。可以输入已有风格，也可以直接输入一个新的风格名。")
        is_correct = False
    else:
        print("未识别的反馈指令，本次跳过。")
        return predictor

    saved_path = save_feedback_image(predictor, image_input, chosen_style)
    log_path = append_feedback_log(
        image_input=image_input,
        predictions=result['predictions'],
        chosen_style=chosen_style,
        saved_path=saved_path,
        is_correct=is_correct
    )

    print(f"\n✓ 反馈已记录")
    print(f"  已归档到: {saved_path}")
    print(f"  日志文件: {log_path}")

    retrain_choice = input("是否立即重建数据并执行快速微调? (y/n): ").strip().lower()
    if retrain_choice == 'y':
        print("\n[反馈闭环] 重新生成训练数据...")
        processor = DataProcessor()
        processor.process_data()
        print("✓ 数据处理完成！")

        print("\n[反馈闭环] 开始快速微调...")
        trainer = Trainer(num_workers_override=0)
        trainer.run_incremental()
        print("✓ 快速微调完成！")

        print("\n[反馈闭环] 重新加载最新模型...")
        predictor = StylePredictor()
        print("✓ 最新模型已加载")

    return predictor


def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     室内设计风格识别系统 - AI Style Recognition System         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("请选择要执行的任务:")
    print("1. 数据处理（从 Excel 中提取数据、创建映射）")
    print("2. 模型训练（使用处理后的数据训练模型）")
    print("3. 图片推理（给定图片识别设计风格）")
    print("4. 完整流程（1 + 2 + 3）")
    print("5. 查看所有可识别的风格")
    print("0. 退出")
    
    choice = input("\n请输入选择 (0-5): ").strip()
    
    if choice == '1':
        print("\n开始数据处理...")
        processor = DataProcessor()
        df, style_to_label, label_to_style = processor.process_data()
        print("✓ 数据处理完成！")
        
    elif choice == '2':
        print("\n开始模型训练...")
        trainer = Trainer()
        trainer.run()
        print("✓ 模型训练完成！")
        
    elif choice == '3':
        print("\n进入图片推理模式...")
        try:
            predictor = StylePredictor()
        except FileNotFoundError as e:
            print(f"{e}")
            sys.exit(1)
        
        while True:
            image_input = input("请输入图片路径或 URL (输入 'quit' 退出): ").strip()
            if image_input.lower() == 'quit':
                break
            
            try:
                result = predictor.predict(image_input, top_k=3, threshold=0.1)
                final_pred = result['final_prediction']
                print("\n最终判断:")
                print(f"  {final_pred['style']}: {final_pred['confidence']:.2%}")
                print("\nTop 3 相似风格（按相似度排序）:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"  {i}. {pred['style']}: {pred['confidence']:.2%}")
                    print(f"     {pred['summary']}")
                
                visualize = input("是否显示图表? (y/n): ").strip().lower()
                if visualize == 'y':
                    predictor.visualize_prediction(result)

                predictor = prompt_feedback_and_maybe_retrain(
                    predictor=predictor,
                    image_input=image_input,
                    result=result
                )
                
            except Exception as e:
                print(f"✗ 预测失败: {e}")
        
    elif choice == '4':
        print("\n开始完整流程...")
        
        # 数据处理
        print("\n[1/3] 数据处理...")
        processor = DataProcessor()
        df, style_to_label, label_to_style = processor.process_data()
        print("✓ 数据处理完成！")
        
        # 模型训练
        print("\n[2/3] 模型训练...")
        trainer = Trainer()
        trainer.run()
        print("✓ 模型训练完成！")
        
        # 推理示例
        print("\n[3/3] 推理示例...")
        predictor = StylePredictor()
        print(f"✓ 可识别 {predictor.num_classes} 种设计风格")
        
    elif choice == '5':
        print("\n正在查看可识别的风格...")
        try:
            predictor = StylePredictor()
        except FileNotFoundError as e:
            print(f"{e}")
            sys.exit(1)
        print(f"\n总共可识别 {predictor.num_classes} 种设计风格:\n")
        
        for idx in range(0, predictor.num_classes, 5):
            row = []
            for i in range(idx, min(idx + 5, predictor.num_classes)):
                row.append(f"{i}: {predictor.label_to_style[i]}")
            print("  |  ".join(row))
        
    elif choice == '0':
        print("退出程序")
        sys.exit(0)
    
    else:
        print("✗ 无效的选择，请重试")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
