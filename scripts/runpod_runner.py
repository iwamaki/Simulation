"""
RunPod統合モジュール

RunPod上でLAMMPSジョブを実行する。
ファイルアップロード → リモート実行 → 結果ダウンロードを1関数で完結。

環境変数:
  RUNPOD_API_KEY   — RunPod APIキー（必須）
  RUNPOD_GPU_TYPE  — GPU種別（デフォルト: "NVIDIA GeForce RTX 3070"）
  RUNPOD_IMAGE     — Dockerイメージ（LAMMPS入り）

使用例:
  from scripts.runpod_runner import run_on_runpod

  run_on_runpod(
      job_dir="/path/to/job",
      input_file="in.deform",
      pot_path="/path/to/potentials/Cu_zhou.eam.alloy",
      np=4,
  )
"""

import os
import re
import sys
import time
import glob
import runpod
import paramiko
import scp as scp_module
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# === 設定 ===

DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 3070"
DEFAULT_IMAGE = "ghcr.io/iwamaki/lammps:latest"
REMOTE_WORK_DIR = "/workspace"
SSH_PROXY_DOMAIN = "ssh.runpod.io"
POD_STARTUP_TIMEOUT = 1200  # Pod起動待ちタイムアウト (秒)
POD_STARTUP_INTERVAL = 10  # Pod起動チェック間隔 (秒)
SSH_CONNECT_TIMEOUT = 120  # SSH接続タイムアウト (秒)
SSH_CONNECT_INTERVAL = 10  # SSH接続リトライ間隔 (秒)
SSH_FALLBACK_PASSWORD = "runpod"  # PUBLIC_KEY未設定時のフォールバック

# ダウンロード対象ファイルのパターン
DOWNLOAD_PATTERNS = [
    "log.lammps",
    "stress_strain.txt",
    "dump.lammpstrj",
    "dump*.lammpstrj",
]


def _get_config():
    """環境変数から設定を取得"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("エラー: RUNPOD_API_KEY 環境変数が設定されていません", file=sys.stderr)
        sys.exit(1)

    return {
        "api_key": api_key,
        "gpu_type": os.environ.get("RUNPOD_GPU_TYPE", DEFAULT_GPU_TYPE),
        "image": os.environ.get("RUNPOD_IMAGE", DEFAULT_IMAGE),
    }


def _create_pod(config):
    """RunPod Podを作成して起動"""
    runpod.api_key = config["api_key"]

    print(f"  Pod作成中... (image={config['image']}, gpu={config['gpu_type']})")

    # ローカルの公開鍵を読み込む
    pub_key = None
    pub_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
    if os.path.exists(pub_key_path):
        with open(pub_key_path, "r") as f:
            pub_key = f.read().strip()
            print(f"  公開鍵を読み込みました: {pub_key_path}")

    create_kwargs = {
        "name": "lammps-job",
        "image_name": config["image"],
        "gpu_type_id": config["gpu_type"],
        "gpu_count": 1,
        "container_disk_in_gb": 10,
        "start_ssh": True,
        # ポート22をTCP公開して直接接続を可能にする（プロキシ回避）
        "ports": "22/tcp", 
        # rootパスワードを設定してstart.shを実行するコマンドを注入
        # SSHD設定でパスワード認証を確実に有効化 (内部のダブルクォートを削除)
        "docker_args": "bash -c 'echo root:runpod | chpasswd && echo PasswordAuthentication yes >> /etc/ssh/sshd_config && /start.sh'"
    }
    if pub_key:
        # 環境変数 PUBLIC_KEY として渡すのが RunPod の仕様
        create_kwargs["env"] = {"PUBLIC_KEY": pub_key}

    pod = runpod.create_pod(**create_kwargs)

    pod_id = pod["id"]
    print(f"  Pod ID: {pod_id}")
    return pod_id


def _wait_for_pod(pod_id, timeout=POD_STARTUP_TIMEOUT):
    """Podが RUNNING 状態になりSSH接続可能になるまで待機"""
    print(f"  Pod起動待ち (最大{timeout}秒)...")
    start = time.time()

    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)

        if pod is None:
            raise RuntimeError(f"Pod {pod_id} が消失しました（イメージ起動失敗の可能性）")

        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime:
            # runtime取得 = Pod起動完了
            # SSH接続先: RunPod TCPプロキシ経由
            ports = runtime.get("ports", [])
            ssh_host = None
            ssh_port = None

            if ports:
                for port_info in ports:
                    if port_info.get("privatePort") == 22:
                        ssh_host = port_info.get("ip")
                        ssh_port = port_info.get("publicPort")
                        break

            if ssh_host and ssh_port:
                print(f"  Pod起動完了: {ssh_host}:{ssh_port} (TCPプロキシ)")
                return ssh_host, int(ssh_port), None
            else:
                # ポート情報がない場合、デバッグ情報を表示
                print(f"  [DEBUG] Runtime ports info: {ports}")
                print(f"  Pod起動完了 (runtime available, ポート情報なし)")
                print(f"  SSHプロキシで接続: {pod_id}@{SSH_PROXY_DOMAIN}")
                return SSH_PROXY_DOMAIN, 22, pod_id

        elapsed = int(time.time() - start)
        print(f"  ... 待機中 ({elapsed}s, status={status})")
        time.sleep(POD_STARTUP_INTERVAL)
    
    # タイムアウト時にログを表示
    print(f"  タイムアウトしました。Podログを取得します: {pod_id}")
    try:
        # ログ取得の試行（APIの仕様によるが、可能な範囲で）
        pass 
    except:
        pass

    raise TimeoutError(f"Pod {pod_id} が{timeout}秒以内に起動しませんでした")


def _find_ssh_key():
    """ローカルのSSH秘密鍵を探す"""
    ssh_dir = os.path.expanduser("~/.ssh")
    # id_ed25519 を優先
    for key_name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
        key_path = os.path.join(ssh_dir, key_name)
        if os.path.exists(key_path):
            return key_path
    return None


def _connect_ssh(host, port, username="root", pod_id=None, timeout=SSH_CONNECT_TIMEOUT):
    """SSH接続を確立（リトライ付き、鍵認証→パスワード認証フォールバック）"""
    # RunPod SSHプロキシの場合、usernameが "{pod_id}" になる
    if pod_id:
        username = pod_id

    print(f"  SSH接続中 ({username}@{host}:{port})...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_key = _find_ssh_key()
    start = time.time()
    last_error = None
    
    # パスワード認証を試行済みかどうかのフラグ
    tried_password = False

    while time.time() - start < timeout:
        try:
            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": 15,
                "banner_timeout": 30,
            }

            # 優先: SSH鍵認証 (まだパスワードフォールバックしていない場合)
            if ssh_key and not tried_password:
                connect_kwargs["key_filename"] = ssh_key
                # print(f"  鍵認証を試行: {ssh_key}")
            else:
                # フォールバック: パスワード認証
                connect_kwargs["password"] = SSH_FALLBACK_PASSWORD
                # print("  パスワード認証を試行")

            client.connect(**connect_kwargs)
            print("  SSH接続成功")
            return client

        except paramiko.ssh_exception.AuthenticationException as e:
            last_error = e
            # 鍵認証で失敗した場合、次はパスワード認証を試すようにフラグを立てる
            if ssh_key and not tried_password:
                print(f"  鍵認証失敗、パスワード認証に切り替えます...")
                tried_password = True
                # 即時リトライ
                continue
            
            elapsed = int(time.time() - start)
            print(f"  ... SSH接続リトライ ({elapsed}s: AuthenticationException)")
            time.sleep(SSH_CONNECT_INTERVAL)

        except (paramiko.ssh_exception.NoValidConnectionsError,
                paramiko.ssh_exception.SSHException,
                OSError) as e:
            last_error = e
            elapsed = int(time.time() - start)
            print(f"  ... SSH接続リトライ ({elapsed}s: {type(e).__name__})")
            time.sleep(SSH_CONNECT_INTERVAL)

    raise ConnectionError(
        f"SSH接続失敗 ({username}@{host}:{port}): {last_error}"
    )


def _ssh_exec(client, command, stream_output=True):
    """SSHコマンド実行（stdout/stderrをリアルタイム表示）"""
    print(f"  実行: {command}")
    stdin, stdout, stderr = client.exec_command(command, timeout=600)

    if stream_output:
        # リアルタイムで出力を表示
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                line = stdout.channel.recv(4096).decode('utf-8', errors='replace')
                sys.stdout.write(line)
                sys.stdout.flush()
            time.sleep(0.1)
        # 残りのデータを読み取り
        remaining = stdout.read().decode('utf-8', errors='replace')
        if remaining:
            sys.stdout.write(remaining)
            sys.stdout.flush()

    exit_code = stdout.channel.recv_exit_status()
    stderr_text = stderr.read().decode('utf-8', errors='replace')

    if stderr_text.strip():
        print(f"  [stderr] {stderr_text.strip()}")

    return exit_code, stderr_text


def _upload_files(client, job_dir, pot_path, remote_dir):
    """ジョブファイルとポテンシャルをPodにアップロード"""
    scp_client = scp_module.SCPClient(client.get_transport())

    # リモートディレクトリ作成
    _ssh_exec(client, f"mkdir -p {remote_dir}", stream_output=False)

    # job_dir内の全ファイルをアップロード
    local_files = glob.glob(os.path.join(job_dir, "*"))
    file_count = 0
    for local_path in local_files:
        if os.path.isfile(local_path):
            fname = os.path.basename(local_path)
            remote_path = f"{remote_dir}/{fname}"
            scp_client.put(local_path, remote_path)
            file_count += 1

    # ポテンシャルファイルをアップロード
    pot_name = os.path.basename(pot_path)
    scp_client.put(pot_path, f"{remote_dir}/{pot_name}")
    file_count += 1

    print(f"  アップロード完了: {file_count}ファイル")
    scp_client.close()
    return pot_name


def _rewrite_potential_path(client, remote_dir, input_file, pot_name):
    """
    LAMMPS入力ファイル内のポテンシャルパスをリモート用に書き換え。

    ローカルの絶対パス → リモートのワーキングディレクトリ内の相対パス
    """
    remote_input = f"{remote_dir}/{input_file}"
    # pair_coeff行のポテンシャルパスを置換
    # pair_coeff * * /absolute/path/to/pot.eam.alloy Element
    #   → pair_coeff * * pot.eam.alloy Element
    sed_cmd = (
        f"sed -i 's|pair_coeff\\(.*\\) [^ ]*/\\([^ /]*\\.eam[^ ]*\\)|"
        f"pair_coeff\\1 {pot_name}|' {remote_input}"
    )
    _ssh_exec(client, sed_cmd, stream_output=False)


def _download_results(client, remote_dir, job_dir):
    """結果ファイルをPodからダウンロード"""
    scp_client = scp_module.SCPClient(client.get_transport())

    # リモートのファイル一覧を取得
    _, stdout, _ = client.exec_command(f"ls {remote_dir}/")
    remote_files = stdout.read().decode().strip().split('\n')

    downloaded = []
    for pattern in DOWNLOAD_PATTERNS:
        for remote_file in remote_files:
            remote_file = remote_file.strip()
            if not remote_file:
                continue
            # シンプルなグロブマッチ
            if _glob_match(pattern, remote_file):
                remote_path = f"{remote_dir}/{remote_file}"
                local_path = os.path.join(job_dir, remote_file)
                try:
                    scp_client.get(remote_path, local_path)
                    downloaded.append(remote_file)
                except scp_module.SCPException:
                    pass

    print(f"  ダウンロード完了: {', '.join(downloaded) if downloaded else 'なし'}")
    scp_client.close()
    return downloaded


def _glob_match(pattern, filename):
    """簡易グロブマッチ（* をワイルドカードとして扱う）"""
    regex = re.escape(pattern).replace(r'\*', '.*')
    return re.fullmatch(regex, filename) is not None


def run_on_runpod(job_dir, input_file, pot_path, np=1, gpu=False, keep_pod=False, pod_id=None):
    """
    RunPod上でLAMMPSジョブを実行する。

    Args:
        job_dir:     ローカルのジョブディレクトリ（入力ファイルが格納されている）
        input_file:  LAMMPS入力ファイル名 (e.g., "in.deform")
        pot_path:    ポテンシャルファイルのローカル絶対パス
        np:          MPI並列数 (default: 1)
        gpu:         GPU使用フラグ (default: False)
        keep_pod:    終了後にPodを残すか (default: False)
        pod_id:      既存のPod IDを使用する場合に指定 (default: None)
    """
    config = _get_config()
    runpod.api_key = config["api_key"]
    
    try:
        # 1. Pod作成・起動 (pod_idが指定されていない場合のみ)
        if not pod_id:
            pod_id = _create_pod(config)
        else:
            print(f"  既存のPodを使用: {pod_id}")

        ssh_info = _wait_for_pod(pod_id)

        # ssh_info は (host, port) または (host, port, pod_id_for_proxy)
        if len(ssh_info) == 3:
            ssh_host, ssh_port, proxy_user = ssh_info
        else:
            ssh_host, ssh_port = ssh_info
            proxy_user = None

        # 2. SSH接続
        client = _connect_ssh(ssh_host, ssh_port, pod_id=proxy_user)

        try:
            remote_dir = f"{REMOTE_WORK_DIR}/job"

            # 3. ファイルアップロード
            print("\n--- ファイル転送 ---")
            pot_name = _upload_files(client, job_dir, pot_path, remote_dir)

            # 4. ポテンシャルパスを書き換え
            _rewrite_potential_path(client, remote_dir, input_file, pot_name)

            # 5. LAMMPS実行
            print("\n--- LAMMPS実行 ---")
            
            # GPUオプションの構築
            gpu_opts = ""
            if gpu:
                # -sf gpu: GPUサフィックスを有効化
                # -pk gpu 1: GPUパッケージを有効化（1 GPU）
                gpu_opts = "-sf gpu -pk gpu 1"

            if np > 1:
                lammps_cmd = f"cd {remote_dir} && mpirun --allow-run-as-root -np {np} lmp {gpu_opts} -in {input_file}"
            else:
                # GPU使用時も標準入力リダイレクトで動作するが、念のためコマンドライン引数形式に統一しても良い
                # ここでは既存のリダイレクト方式にオプションを追加
                lammps_cmd = f"cd {remote_dir} && lmp {gpu_opts} -in {input_file}" # standard input redirection removed for clarity if gpu opts are complex, but keeping simple for now. 
                # Actually, 'lmp -sf gpu -in in.file' is safer than 'lmp -sf gpu < in.file' depending on lammps version, 
                # but let's stick to the pattern.
                lammps_cmd = f"cd {remote_dir} && lmp {gpu_opts} -in {input_file}"
            
            lammps_cmd += " > log.lammps 2>&1"

            exit_code, stderr = _ssh_exec(client, lammps_cmd, stream_output=True)

            if exit_code == 0:
                print("\n  LAMMPS実行完了")
            else:
                print(f"\n  LAMMPS実行失敗 (exit_code={exit_code})")
                # log.lammpsの末尾を表示
                _ssh_exec(client, f"tail -n 20 {remote_dir}/log.lammps")

            # 6. 結果ダウンロード
            print("\n--- 結果ダウンロード ---")
            _download_results(client, remote_dir, job_dir)

        finally:
            client.close()

    finally:
        # 7. Pod停止（課金を止める）
        if pod_id and not keep_pod:
            print(f"\n  Pod終了中... ({pod_id})")
            try:
                runpod.terminate_pod(pod_id)
                print("  Pod終了完了")
            except Exception as e:
                print(f"  Pod終了エラー: {e}")
                print(f"  手動で終了してください: runpod.terminate_pod('{pod_id}')")
        elif pod_id and keep_pod:
            print(f"\n  Pod維持: {pod_id} (--keep-pod)")
            print(f"  手動停止: runpod.terminate_pod('{pod_id}')")
