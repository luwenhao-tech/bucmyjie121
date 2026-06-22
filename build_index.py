#!/usr/bin/env python3
"""构建论文向量索引。

使用方法：
    1. 将论文 PDF 放入 papers/ 目录
    2. 运行：python build_index.py
    3. 强制重建（忽略缓存）：python build_index.py --force
    4. 指定目录：python build_index.py --dir /path/to/papers

索引数据存储在 chroma_db/ 目录，服务启动时自动加载。
"""
import sys
import argparse
from rag_engine import build_index, get_index_stats, ensure_dirs, PAPERS_DIR


def main():
    parser = argparse.ArgumentParser(description="构建论文 PDF 向量索引")
    parser.add_argument("--dir", default=PAPERS_DIR, help=f"论文 PDF 目录（默认: {PAPERS_DIR}）")
    parser.add_argument("--force", action="store_true", help="强制重建所有索引（忽略缓存）")
    parser.add_argument("--stats", action="store_true", help="仅显示当前索引状态")
    args = parser.parse_args()

    ensure_dirs()

    if args.stats:
        stats = get_index_stats()
        print(f"\n📊 索引状态：")
        print(f"   状态: {stats['status']}")
        print(f"   文件数: {stats['total_files']}")
        print(f"   文本块数: {stats['total_chunks']}")
        if stats['files']:
            print(f"   已索引文件:")
            for f in stats['files']:
                print(f"     - {f}")
        print()
        return

    print(f"\n📚 论文索引构建工具")
    print(f"   论文目录: {args.dir}")
    print(f"   强制重建: {'是' if args.force else '否'}")

    results = build_index(papers_dir=args.dir, force=args.force)

    if not results:
        print("\n💡 提示：请将论文 PDF 文件放入 papers/ 目录后重新运行")
        print("   mkdir -p papers")
        print("   cp /path/to/your/papers/*.pdf papers/")
        print("   python build_index.py\n")
        sys.exit(1)

    # 汇总
    success = sum(1 for v in results.values() if v >= 0)
    failed = sum(1 for v in results.values() if v < 0)
    new_chunks = sum(v for v in results.values() if v > 0)
    skipped = sum(1 for v in results.values() if v == 0)

    print(f"\n📋 结果汇总：")
    print(f"   成功: {success} 个文件（其中新索引 {success - skipped} 个，跳过 {skipped} 个）")
    if failed:
        print(f"   失败: {failed} 个文件")
    print(f"   新增文本块: {new_chunks}")
    print()

    # 显示最终统计
    stats = get_index_stats()
    print(f"📊 数据库总量：{stats['total_chunks']} 个文本块，来自 {stats['total_files']} 个文件\n")


if __name__ == "__main__":
    main()
