#!/usr/bin/env python3
import os
import re
import subprocess

def install_deps_from_file(deps_file):
    if not os.path.exists(deps_file):
        return
    print(f">>> [Installer] Processing {deps_file}...")
    with open(deps_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Case 1: GitHub Zip Archive URL (e.g. repo @ https://github.com/owner/repo/archive/commit.zip)
            m_zip = re.search(r"https://github\.com/([^/]+)/([^/]+)/archive/(.+)\.zip", line)
            if m_zip:
                owner = m_zip.group(1)
                repo = m_zip.group(2)
                commit = m_zip.group(3)
                git_url = f"https://github.com/{owner}/{repo}.git"
                local_dir = f"/tmp/{repo}"
                print(f">>> [Installer] Git cloning {git_url} (ref: {commit}) -> {local_dir}...")
                if os.path.exists(local_dir):
                    subprocess.run(["rm", "-rf", local_dir], check=True)
                subprocess.run(["git", "clone", "--filter=blob:none", git_url, local_dir], check=True)
                subprocess.run(["git", "-C", local_dir, "checkout", commit], check=True)
                print(f">>> [Installer] Installing {local_dir}...")
                subprocess.run(["pip", "install", local_dir, "--no-deps", "--no-build-isolation"], check=True)
                continue

            # Case 2: Git Repository URL (e.g. vllm @ git+https://github.com/vllm-project/vllm@commit)
            m_git = re.search(r"git\+(https://github\.com/([^/]+)/([^/@]+))@([a-f0-9]+)", line)
            if m_git:
                base_git_url = m_git.group(1)
                repo_name = m_git.group(3)
                git_sha = m_git.group(4)
                git_url = base_git_url if base_git_url.endswith(".git") else f"{base_git_url}.git"
                local_dir = f"/tmp/{repo_name}"
                print(f">>> [Installer] Git cloning {git_url} (ref: {git_sha}) -> {local_dir}...")
                if os.path.exists(local_dir):
                    subprocess.run(["rm", "-rf", local_dir], check=True)
                subprocess.run(["git", "clone", "--filter=blob:none", git_url, local_dir], check=True)
                subprocess.run(["git", "-C", local_dir, "checkout", git_sha], check=True)
                print(f">>> [Installer] Installing {local_dir}...")
                subprocess.run(["pip", "install", local_dir, "--no-deps", "--no-build-isolation"], check=True)
                continue

            # Fallback
            print(f">>> [Installer] Installing fallback {line}...")
            subprocess.run(["pip", "install", line, "--no-deps", "--no-build-isolation"], check=True)

# Install pre-train and post-train GitHub dependencies via git clone
install_deps_from_file("/deps/src/dependencies/extra_deps/pre_train_github_deps.txt")
install_deps_from_file("/deps/src/dependencies/extra_deps/post_train_github_deps.txt")

print(">>> [Installer] Installing MaxText vLLM adapter...")
adapter_path = "/deps/src/maxtext/integration/vllm"
if os.path.exists(adapter_path):
    subprocess.run(["pip", "install", adapter_path, "--no-deps", "--no-build-isolation"], check=True)

print(">>> [Installer] Ensuring GPU-only triton is completely removed...")
subprocess.run(["pip", "uninstall", "-y", "triton"], check=False)
print(">>> [Installer] Dependencies installation complete.")
