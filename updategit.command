#!/bin/bash
cd "$(dirname "$0")"
git commit --amend -m "update gitignore"
git push --force
echo ""
echo "Feito! Podes fechar esta janela."
read -p "Pressiona Enter para fechar..."
