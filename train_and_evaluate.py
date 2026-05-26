import sys
import os
import subprocess
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_training():
    print("="*70)
    print("开始训练...")
    print("="*70)
    
    cmd = [sys.executable, "train_2000_episodes.py", "--resume"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    return process

def wait_for_training_completion(process):
    print("\n等待训练完成...")
    print("="*70)
    
    while True:
        if process.poll() is not None:
            break
        time.sleep(60)
    
    return process.returncode

def run_evaluation():
    print("\n" + "="*70)
    print("训练完成！开始评估最终模型...")
    print("="*70)
    
    cmd = [
        sys.executable, "evaluate_final_model.py",
        "--episodes", "100",
        "--compare"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("错误信息:")
        print(result.stderr)
    
    return result.returncode

def main():
    print("="*70)
    print("麻将AI - 训练与评估自动化流程")
    print("="*70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    training_process = run_training()
    
    try:
        return_code = wait_for_training_completion(training_process)
        
        if return_code == 0:
            print("\n✅ 训练成功完成！")
            eval_return_code = run_evaluation()
            
            if eval_return_code == 0:
                print("\n✅ 评估完成！")
            else:
                print(f"\n❌ 评估失败，返回码: {eval_return_code}")
        else:
            print(f"\n❌ 训练失败，返回码: {return_code}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在停止训练...")
        training_process.terminate()
        try:
            training_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            training_process.kill()
        print("训练已停止")
    
    print("\n" + "="*70)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
