#!/bin/bash
cd "$(dirname "$0")"
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
git push
echo ""
echo "Feito! Podes fechar esta janela."
read -p "Pressiona Enter para fechar..."
