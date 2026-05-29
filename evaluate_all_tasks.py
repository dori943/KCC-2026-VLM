import os
import shutil
import subprocess
import sys
import time

def run_multi_task_evaluation():
    task_configs = {
        "Task 1 (Balloon)": {
            "script": "main_simulation_balloon.py",
            "target_json": "module3_balloon_output.json",
            "source_json": "task1_balloon_origin.json"
        },
        "Task 2 (Chain)": {
            "script": "main_simulation_chain.py",
            "target_json": "module3_chain_output.json",
            "source_json": "task2_chain_origin.json"
        },
        "Task 3 (Pet)": {
            "script": "main_simulation_pet.py",
            "target_json": "module3_pet_output.json",
            "source_json": "task3_pet_origin.json"
        }
    }
    
    # 💡 꿀팁: 한 판에 4분씩 걸린다면, 우선 제대로 도는지 확인하기 위해 
    # trials 횟수를 3회나 5회로 줄여서 테스트해보는 것을 강력히 권장하네!
    num_trials = 20 
    
    print("=" * 70)
    print(f"▶ [실시간 로그 스트리밍 모드] 조립 성공률 자동 평가 시작")
    print("=" * 70)
    
    results_summary = {}
    
    for task_name, config in task_configs.items():
        script_file = config["script"]
        target_json = config["target_json"]
        source_json = config["source_json"]
        
        if not os.path.exists(script_file) or not os.path.exists(source_json):
            print(f"[경고] {task_name} 파일 누락으로 스킵합니다.")
            continue
            
        print(f"\n[진행] {task_name} ➔ {script_file} 실행 시작...")
        success_count = 0
        
        for trial in range(1, num_trials + 1):
            print(f"\n  ================ [ 시도 {trial:02d}/{num_trials:02d} ] ================")
            shutil.copy(source_json, target_json)
            
            custom_env = os.environ.copy()
            custom_env["ENABLE_AFFORDANCE_R1"] = "1"
            custom_env["ENABLE_SAM2_REFINEMENT"] = "1"

            
            # Popen을 사용하여 실시간 출력을 가로챕니다.
            process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # 에러 로그도 stdout으로 통합해서 받음
                text=True,
                env=custom_env
            )
            
            success_in_trial = False
            
            # 서브프로세스의 로그를 한 줄씩 실시간으로 읽어와 터미널에 출력
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    clean_line = line.strip()
                    print(f"    │ [Sim Log] {clean_line}") # 시뮬레이션 내부 출력을 실시간 표기
                    
                    if "assembly successful" in clean_line:
                        success_in_trial = True
            
            # 최종 종료 상태 확인
            process.wait()
            
            if success_in_trial:
                print("  ➔ 결과: 성공 (Success)")
                success_count += 1
            else:
                print("  ➔ 결과: 실패 (Fail)")
                
        success_rate = (success_count / num_trials) * 100
        results_summary[task_name] = {"success": success_count, "total": num_trials, "rate": success_rate}
        print(f"\n▷ {task_name} 최종: {success_count}/{num_trials} 성공 ({success_rate:.1f}%)")

    print("\n" + "=" * 70)
    print("                    [ 3개 태스크 최종 평가 리포트 ]")
    print("=" * 70)
    for task_name, res in results_summary.items():
        print(f" * {task_name:<25}: {res['success']:02d}/{res['total']:02d} 성공 ({res['rate']:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_multi_task_evaluation()