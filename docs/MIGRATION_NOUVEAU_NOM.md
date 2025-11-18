# Migration - Nouveau Nom du Repository

**Le repo a été renommé de `Agentic-RAG` → `RAG-Fusion`**

---

## 🔄 Si tu as DÉJÀ cloné l'ancienne version

### Option 1 : Mettre à jour le remote (Rapide)

```bash
# Aller dans ton dossier local
cd Agentic-RAG  # ou le chemin où tu as cloné

# Mettre à jour le remote
git remote set-url origin https://github.com/gillesturpin/RAG-Fusion.git

# Vérifier que c'est bon
git remote -v

# Pull les derniers changements
git pull origin main
```

✅ **C'est tout !** Ton dossier local peut garder l'ancien nom `Agentic-RAG`, ça ne pose aucun problème.

---

### Option 2 : Cloner à nouveau (Propre)

Si tu préfères repartir de zéro :

```bash
# Sauvegarder ton .env si tu en as un
cp Agentic-RAG/.env ~/backup.env

# Supprimer l'ancien
rm -rf Agentic-RAG

# Cloner le nouveau
git clone https://github.com/gillesturpin/RAG-Fusion.git
cd RAG-Fusion

# Restaurer ton .env
cp ~/backup.env .env
```

---

## 🆕 Si tu n'as PAS encore cloné

Utilise directement le nouveau nom :

```bash
git clone https://github.com/gillesturpin/RAG-Fusion.git
cd RAG-Fusion
```

---

## ❓ Pourquoi le changement ?

Le projet a évolué :
- **Avant** : Architecture agentique avec LangGraph
- **Maintenant** : Architecture simplifiée avec RAG Fusion (simple chains)

Le nouveau nom `RAG-Fusion` reflète mieux l'implémentation actuelle.

---

## 🔗 Redirection Automatique

**Bonne nouvelle** : GitHub redirige automatiquement l'ancien nom vers le nouveau !

Donc même `https://github.com/gillesturpin/Agentic-RAG` fonctionne encore et redirige vers `RAG-Fusion`.

Mais il vaut mieux mettre à jour pour éviter les confusions futures.

---

## ✅ Vérifier que tout fonctionne

Après la mise à jour :

```bash
# Vérifier le remote
git remote -v
# Doit afficher : https://github.com/gillesturpin/RAG-Fusion.git

# Pull les derniers changements
git pull origin main

# Vérifier le README
head -3 README.md
# Doit afficher : # RAG Fusion
```

---

**Questions ?** Contacte l'équipe ou consulte `docs/QUICK_START_COLLEGUES.md`
