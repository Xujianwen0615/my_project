# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from semilearn.core.utils import get_net_builder
import warnings
warnings.filterwarnings('ignore')

# ===================== AutoDL 中文/负号显示修复 + 可视化优化 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 创建可视化保存目录，避免报错
os.makedirs("./pred_visualization", exist_ok=True)

def num2class(num):
    """数字标签 转 class_X 格式的字符串标签，核心转换函数"""
    return f"class_{num}"

def predict_single_image(net, img_path, transform, device, num_classes):
    """单张图片预测核心函数，返回：原始图、所有概率、数字预测标签、class格式预测标签"""
    raw_img = Image.open(img_path).convert('RGB')
    img_tensor = transform(raw_img).unsqueeze(0)
    img_tensor = img_tensor.type(torch.FloatTensor).to(device)
    
    all_probs = None
    pred_label_num = None
    with torch.no_grad():
        feat = net(img_tensor, only_feat=True)
        logit = net(feat, only_fc=True)
        all_probs = logit.softmax(dim=-1).cpu().numpy()[0]
        pred_label_num = np.argmax(all_probs)  # 模型输出的数字标签
    pred_label_class = num2class(pred_label_num) # 转为 class_X 格式
    return raw_img, all_probs, pred_label_num, pred_label_class

def visualize_and_save(raw_img, all_probs, pred_label_class, save_name, num_classes):
    """可视化并保存图片到指定路径，标题显示class_X格式"""
    plt.figure(figsize=(12, 5))
    # 子图1：原始图像 + class格式预测标签
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    pred_prob = all_probs[int(pred_label_class.split('_')[-1])]
    plt.title(f'Pred Label: {pred_label_class} | Prob: {pred_prob:.4f}', fontsize=12)
    plt.axis('off')

    # 子图2：概率柱状图（X轴保留数字，因为类别顺序是0-14，更直观）
    plt.subplot(1, 2, 2)
    cls_index = np.arange(num_classes)
    bars = plt.bar(cls_index, all_probs, color='lightskyblue', alpha=0.8)
    pred_num = int(pred_label_class.split('_')[-1])
    bars[pred_num].set_color('crimson')
    bars[pred_num].set_alpha(1)

    # 给每个柱子标注概率值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{all_probs[i]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Class Index (0~14)', fontsize=11)
    plt.ylabel('Pred Probability ', fontsize=11)
    plt.title(f'All Class Probability Distribution\nMax Prob: {pred_prob:.4f}', fontsize=12)
    plt.xticks(cls_index)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭画布，避免内存泄漏

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True, help='pth权重文件路径')
    parser.add_argument('--img_path', type=str, default=None, help='待预测的单张图像路径【二选一】')
    parser.add_argument('--img_dir', type=str, default=None, help='待预测的图片根文件夹路径【二选一】，格式：根文件夹/class_0/图片.png')

    '''
    Backbone Net Configurations 与你的代码完全一致
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations 完全保留你的配置，无任何改动
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()

    # ===================== 1. 加载权重+构建模型 【完全复刻你的代码，一字未改】 =====================
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''

    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()

    # ===================== 2. 图像预处理【完全保留你的配置，custom数据集均值方差不变】 =====================
    mean_std = {
        'cifar10': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
        'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'svhn': [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)],
        'custom':[(0.232, 0.227, 0.276), (0.208, 0.222, 0.226)]  # 你的自定义数据集均值方差
    }
    mean, std = mean_std[args.dataset]

    crop_size = int(args.img_size * args.crop_ratio)
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(crop_size),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # ===================== 3. 双模式：单张图片预测 OR 批量文件夹预测 =====================
    img_suffix = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    
    if args.img_path is not None and os.path.exists(args.img_path):
        # ========== 模式1：单张图片预测（全部显示class_X格式） ==========
        raw_img, all_probs, pred_label_num, pred_label_class = predict_single_image(net, args.img_path, transform, device, args.num_classes)
        # 打印概率
        print("=" * 60)
        print(f"数据集: {args.dataset} | 类别总数: {args.num_classes}")
        print(f"最终预测类别: {pred_label_class} | 该类别概率: {all_probs[pred_label_num]:.6f}")
        print("=" * 60)
        print("【所有类别的预测概率值】")
        for cls_idx in range(args.num_classes):
            print(f"类别 {num2class(cls_idx)} : {all_probs[cls_idx]:.6f}")
        print("=" * 60)
        # 保存可视化
        save_img_path = "./pred_visualization/single_pred_result.png"
        visualize_and_save(raw_img, all_probs, pred_label_class, save_img_path, args.num_classes)
        print(f"\n✅ 可视化结果已保存至: {save_img_path}")

    elif args.img_dir is not None and os.path.exists(args.img_dir):
        # ========== 模式2：批量文件夹预测 + 准确率计算 ✅【核心适配：真实标签=class_X格式】 ==========
        total_num = 0  # 总图片数
        correct_num = 0  # 预测正确的图片数
        print("="*80)
        print(f"开始批量预测文件夹: {args.img_dir}")
        print(f"数据集类别数: {args.num_classes} | 设备: {device}")
        print(f"✅ 真实标签格式: class_0 ~ class_14")
        print(f"✅ 预测标签格式: class_0 ~ class_14")
        print("="*80)
        
        # 遍历文件夹：根目录/class_0/图片文件 → 真实标签直接等于文件夹名 class_0
        for true_label_class in os.listdir(args.img_dir):
            true_label_path = os.path.join(args.img_dir, true_label_class)
            if not os.path.isdir(true_label_path):
                continue  # 跳过文件，只处理子文件夹
            
            # 只处理 class_X 格式的文件夹，其他格式自动跳过
            if not true_label_class.startswith("class_"):
                print(f"⚠️  跳过非class_X格式文件夹: {true_label_class}")
                continue
            
            # 遍历当前类别下的所有图片
            for img_name in os.listdir(true_label_path):
                img_path = os.path.join(true_label_path, img_name)
                if os.path.splitext(img_name)[-1].lower() not in img_suffix:
                    continue
                total_num += 1
                
                # 单张图片预测
                raw_img, all_probs, pred_label_num, pred_label_class = predict_single_image(net, img_path, transform, device, args.num_classes)
                
                # ✅ 核心判断：字符串直接匹配 class_X == class_X
                is_correct = (pred_label_class == true_label_class)
                if is_correct:
                    correct_num += 1
                
                # ✅ 控制台打印：全部显示class_X格式，直观易懂
                pred_prob = all_probs[pred_label_num]
                print(f"[{total_num}] 图片: {img_name} | 真实标签: {true_label_class} | 预测标签: {pred_label_class} | 结果: {'✅正确' if is_correct else '❌错误'} | 置信度: {pred_prob:.4f}")
                
                # 保存可视化图片，命名包含【真实class_*_预测class_*】
                save_img_name = f"{true_label_class}_pred_{pred_label_class}_{img_name}"
                save_img_path = os.path.join("./pred_visualization", save_img_name)
                visualize_and_save(raw_img, all_probs, pred_label_class, save_img_path, args.num_classes)
        
        # ========== 计算并输出最终准确率 ==========
        if total_num > 0:
            acc = correct_num / total_num
            print("="*80)
            print("📊 批量预测结果统计【class_X格式】")
            print("="*80)
            print(f"总预测图片数量: {total_num}")
            print(f"预测正确数量  : {correct_num}")
            print(f"预测错误数量  : {total_num - correct_num}")
            print(f"📈 整体预测准确率: {acc * 100:.4f}%")
            print("="*80)
            print(f"✅ 所有可视化结果已保存至: ./pred_visualization")
        else:
            print("❌ 文件夹内未找到有效图片或有效class_X格式文件夹")
    
    else:
        print("❌ 请指定有效路径：--img_path 单张图片路径  或者  --img_dir 批量文件夹路径")