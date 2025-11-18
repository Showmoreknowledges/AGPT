import argparse
import numpy as np
import os

def merge_pairs_inplace(npz_path, output_key="aligned_pair"):
    """
    在原始 .npz 文件中将 pos_pair 与 test_pair 合并为 aligned_pair，并覆盖保存。

    参数：
        npz_path (str): 输入数据集路径，例如 './douban.npz'
        output_key (str): 合并后的键名（默认 'aligned_pair'）
    """
    # 1️⃣ 读取 npz
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"✅ Loaded {npz_path}. Keys: {keys}")

    # 2️⃣ 自动检测 pair 键名
    pos_key = next((k for k in keys if "pos" in k), None)
    test_key = next((k for k in keys if "test" in k), None)

    if pos_key is None or test_key is None:
        raise KeyError(f"❌ 在 {npz_path} 中未找到 pos_pair 与 test_pair 键，请检查文件内容。")

    # 3️⃣ 合并
    pos_pairs = data[pos_key]
    test_pairs = data[test_key]
    aligned_pairs = np.concatenate([pos_pairs, test_pairs], axis=0)
    print(f"🔗 Merged '{pos_key}' + '{test_key}' → '{output_key}', shape = {aligned_pairs.shape}")

    # 4️⃣ 构建新的字典（去掉原 pos/test，加入 aligned）
    new_data = {k: data[k] for k in data.files if k not in [pos_key, test_key]}
    new_data[output_key] = aligned_pairs

    # 5️⃣ 覆盖保存为原文件
    np.savez(npz_path, **new_data)
    print(f"💾 Updated file saved (in-place): {npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge pos_pair and test_pair into aligned_pair (in-place in .npz)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .npz file")
    parser.add_argument("--output_key", type=str, default="aligned_pair", help="Key name for merged pairs in npz")
    args = parser.parse_args()

    merge_pairs_inplace(args.data, args.output_key)


if __name__ == "__main__":
    main()
