import os

def has_author(f):
  for line in f:
    line = line.split()
    if '@author' in line:
      return True
    if 'class' in line or 'interface' in line:
      return False
  assert False, 'No class found...'

for (folder,dirs,files) in os.walk("."):
  for fn in files:
    fn = '%s/%s' % (folder, fn)
    if fn.endswith('.java'):
      with open(fn) as f:
        if not has_author(f):
          print fn

