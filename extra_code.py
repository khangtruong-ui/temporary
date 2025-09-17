with open('vit-rvsa/install.py') as f:
    content = f.read()

with open('vit-rvsa/install.py', 'w') as f:
    new_content = content.replace('use_abs_pos_emb=True', 'use_abs_pos_emb=False')
    f.write(new_content)
