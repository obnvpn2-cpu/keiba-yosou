#!/bin/bash
#
# Git初期化とプッシュのクイックスタートスクリプト
# 
# 使い方:
# 1. REPO_URLを自分のリポジトリURLに変更
# 2. chmod +x git_push.sh
# 3. ./git_push.sh
#

# ========================================
# ⚠️ ここを自分のリポジトリURLに変更してください
# ========================================
REPO_URL="https://github.com/obnvpn2-cpu/keiba-yosou.git"

# ========================================
# 以下は変更不要
# ========================================

echo "========================================="
echo "Git初期化とプッシュを開始します"
echo "========================================="
echo ""

# リポジトリURLが変更されているか確認
if [ "$REPO_URL" = "https://github.com/your-username/keiba-scenario-ai.git" ]; then
    echo "⚠️  エラー: REPO_URLを自分のリポジトリURLに変更してください"
    echo "このスクリプトの上部でREPO_URLを編集してください"
    exit 1
fi

# Step 1: Git初期化
echo "Step 1: Gitリポジトリを初期化..."
git init
echo "✅ Git初期化完了"
echo ""

# Step 2: ファイルをステージング
echo "Step 2: 全ファイルをステージング..."
git add .
echo "✅ ステージング完了"
echo ""

# Step 3: 最初のコミット
echo "Step 3: 最初のコミット..."
git commit -m "Initial commit: Phase 1 implementation complete

- Timeline manager (walk-forward CV, leak prevention)
- Base model (LightGBM + calibration)
- Scenario adjustment models (Baba, Pace)
- Probability integration (log-odds + Softmax)
- Backtest engine (odds timing, takeout rate)
- SHAP explainer + text generation
- FastAPI + MLflow integration
- Complete documentation"
echo "✅ コミット完了"
echo ""

# Step 4: ブランチ名をmainに設定
echo "Step 4: メインブランチ名を設定..."
git branch -M main
echo "✅ ブランチ名設定完了"
echo ""

# Step 5: リモートリポジトリを追加
echo "Step 5: リモートリポジトリを追加..."
git remote add origin "$REPO_URL"
echo "✅ リモートリポジトリ追加完了"
echo ""

# Step 6: プッシュ
echo "Step 6: リモートにプッシュ..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 成功！すべての手順が完了しました"
    echo "========================================="
    echo ""
    echo "リポジトリURL: ${REPO_URL%.git}"
else
    echo ""
    echo "========================================="
    echo "⚠️  プッシュ中にエラーが発生しました"
    echo "========================================="
    echo ""
    echo "考えられる原因:"
    echo "1. リモートリポジトリが存在しない"
    echo "2. 認証に失敗した"
    echo "3. リモートに既にコミットが存在する"
    echo ""
    echo "手動で確認してください:"
    echo "  git remote -v"
    echo "  git pull origin main --rebase"
    echo "  git push origin main"
fi
