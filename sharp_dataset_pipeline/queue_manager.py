import os
import json
import argparse
import sys
from typing import Dict, Any, Optional

def main():
    parser = argparse.ArgumentParser(description="Sharp Dataset Pipeline 运行时队列管理工具")
    parser.add_argument("--save-dir", required=True, help="流水线运行目录")
    parser.add_argument("--action", choices=["add", "clear", "list"], default="list", help="执行操作")
    parser.add_argument("--image-id", help="图片ID (仅 add 操作需要)")
    parser.add_argument("--image-path", help="本地图片路径 (可选)")
    parser.add_argument("--download-url", help="下载地址 (可选)")
    parser.add_argument("--hf-upload", type=str, choices=["1", "0", "true", "false"], default="true", help="是否上传到 HF")
    
    args = parser.parse_args()
    
    queue_file = os.path.join(args.save_dir, "pending_queue.jsonl")
    
    if args.action == "add":
        if not args.image_id:
            print("错误: add 操作必须提供 --image-id")
            sys.exit(1)
            
        should_upload = args.hf_upload.lower() in ("1", "true")
        
        task = {
            "image_id": args.image_id,
            "image_path": args.image_path or "",
            "download_location": args.download_url or "",
            "hf_upload": should_upload,
            "manual": True
        }
        
        os.makedirs(args.save_dir, exist_ok=True)
        with open(queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"成功添加任务: {args.image_id} (hf_upload={should_upload})")
        
    elif args.action == "clear":
        if os.path.exists(queue_file):
            os.remove(queue_file)
            print("队列文件已清理")
        else:
            print("队列文件不存在")
            
    elif args.action == "list":
        if not os.path.exists(queue_file):
            print("当前队列为空")
            return
            
        count = 0
        with open(queue_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
                    try:
                        t = json.loads(line)
                        print(f"[{count}] ID: {t.get('image_id')} | HF: {t.get('hf_upload')} | Manual: {t.get('manual', False)}")
                    except:
                        pass
        print(f"\n总计 {count} 个待处理任务")

if __name__ == "__main__":
    main()
