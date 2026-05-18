import sys
import os
import json
import http.server
import socketserver
import threading
import time
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainingMonitor:
    def __init__(self, project_root):
        self.project_root = project_root
        self.current_episode = 0
        self.total_episodes = 0
        self.is_training = False
        self.start_time = None
        self.episode_rewards = deque(maxlen=100)
        self.evaluation_results = []
        self._lock = threading.Lock()
        self._status_file = None
        self._last_status_mtime = 0
        
    def set_status_file(self, log_dir):
        self._status_file = os.path.join(self.project_root, log_dir, 'realtime_status.json')
        
    def check_status_file(self):
        if self._status_file and os.path.exists(self._status_file):
            try:
                mtime = os.path.getmtime(self._status_file)
                if mtime > self._last_status_mtime:
                    self._last_status_mtime = mtime
                    with open(self._status_file, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        self._update_from_status(status)
            except Exception as e:
                pass
    
    def _update_from_status(self, status):
        with self._lock:
            self.current_episode = status.get('current_episode', 0)
            self.total_episodes = status.get('total_episodes', 0)
            self.is_training = status.get('is_training', False)
            
            if status.get('start_time') and not self.start_time:
                self.start_time = datetime.fromisoformat(status['start_time'])
            
            recent_rewards = status.get('recent_rewards', [])
            for r in recent_rewards:
                self.episode_rewards.append(r)
            
            eval_results = status.get('evaluation_results', [])
            for eval_result in eval_results:
                exists = any(e['episode'] == eval_result.get('episode') for e in self.evaluation_results)
                if not exists:
                    self.evaluation_results.append(eval_result)
    
    def update_status(self, episode, total_episodes, reward=None):
        with self._lock:
            self.current_episode = episode
            self.total_episodes = total_episodes
            if self.start_time is None:
                self.start_time = datetime.now()
            self.is_training = True
            if reward is not None:
                self.episode_rewards.append(reward)
    
    def add_evaluation(self, episode, win_rates, avg_rewards):
        with self._lock:
            self.evaluation_results.append({
                'episode': episode,
                'win_rates': win_rates,
                'avg_rewards': avg_rewards,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_status(self):
        self.check_status_file()
        
        with self._lock:
            elapsed = 0
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
            
            progress = 0
            if self.total_episodes > 0:
                progress = (self.current_episode / self.total_episodes) * 100
            
            avg_reward = 0
            if len(self.episode_rewards) > 0:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            
            return {
                'current_episode': self.current_episode,
                'total_episodes': self.total_episodes,
                'progress': progress,
                'is_training': self.is_training,
                'elapsed_seconds': elapsed,
                'avg_reward': avg_reward,
                'recent_rewards': list(self.episode_rewards)[-20:],
                'evaluation_results': self.evaluation_results[-10:],
                'start_time': self.start_time.isoformat() if self.start_time else None
            }

class MahjongHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    monitor = None
    project_root = None
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == '/api/status':
            self.send_json(self.monitor.get_status())
        elif path == '/api/history':
            version = query.get('version', ['v3'])[0]
            self.send_json(self.get_training_history(version))
        elif path == '/api/checkpoints':
            version = query.get('version', ['v3'])[0]
            self.send_json(self.get_checkpoints(version))
        elif path == '/api/versions':
            self.send_json(self.get_all_versions())
        else:
            super().do_GET()
    
    def get_training_history(self, version='v3'):
        log_dir = os.path.join(self.project_root, f'logs_{version}' if version != 'v1' else 'logs')
        if not os.path.exists(log_dir):
            return {'error': 'Log directory not found'}
        
        history_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.json') and f != 'realtime_status.json'])
        if not history_files:
            return {'error': 'No history files found'}
        
        latest_file = os.path.join(log_dir, history_files[-1])
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {'error': str(e)}
    
    def get_checkpoints(self, version='v3'):
        checkpoint_dir = os.path.join(self.project_root, f'checkpoints_{version}' if version != 'v1' else 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            return []
        
        checkpoints = []
        for f in sorted(os.listdir(checkpoint_dir)):
            if f.endswith('.pt') and 'agent0' in f:
                filepath = os.path.join(checkpoint_dir, f)
                stat = os.stat(filepath)
                checkpoints.append({
                    'name': f,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        return checkpoints
    
    def get_all_versions(self):
        versions = []
        for v in ['v1', 'v2', 'v3', 'realtime']:
            log_dir = os.path.join(self.project_root, f'logs_{v}' if v != 'v1' else 'logs')
            checkpoint_dir = os.path.join(self.project_root, f'checkpoints_{v}' if v != 'v1' else 'checkpoints')
            
            if os.path.exists(log_dir) or os.path.exists(checkpoint_dir):
                versions.append({
                    'version': v,
                    'has_logs': os.path.exists(log_dir),
                    'has_checkpoints': os.path.exists(checkpoint_dir),
                    'checkpoint_count': len([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]) if os.path.exists(checkpoint_dir) else 0
                })
        return versions
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

def main():
    port = 8081
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
    os.makedirs(web_dir, exist_ok=True)
    
    monitor = TrainingMonitor(project_root)
    monitor.set_status_file('logs_realtime')
    MahjongHTTPRequestHandler.monitor = monitor
    MahjongHTTPRequestHandler.project_root = project_root
    
    os.chdir(web_dir)
    
    handler = MahjongHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🀄 麻将AI Web 服务器已启动")
        print(f"访问地址: http://localhost:{port}")
        print(f"项目根目录: {project_root}")
        print(f"监控日志目录: {os.path.join(project_root, 'logs_realtime')}")
        print(f"实时监控已启用")
        print(f"按 Ctrl+C 停止服务器")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    main()
