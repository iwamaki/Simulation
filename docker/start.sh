#!/bin/bash
# RunPod用スタートアップスクリプト
# PUBLIC_KEY環境変数からSSH公開鍵を設定し、sshdを起動する

# SSHホスト鍵の生成（初回起動時）
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -N ''
    ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -N ''
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -N ''
fi

# RunPodが注入するPUBLIC_KEY環境変数からauthorized_keysを設定
mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "$PUBLIC_KEY" ] && [ "$PUBLIC_KEY" != "null" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "SSH公開鍵を設定しました"
fi

# sshdをフォアグラウンドで起動
echo "SSHサーバー起動中..."
exec /usr/sbin/sshd -D
