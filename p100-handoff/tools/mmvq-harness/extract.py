# Pull per-GPU q6_K weight slices out of the gguf, as the tensor split gives them to each GPU.
import struct, sys
P='/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf'
f=open(P,'rb')
magic,ver,nt,nkv=struct.unpack('<4sIQQ',f.read(24))
def s(): n,=struct.unpack('<Q',f.read(8)); return f.read(n)
sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip(t):
    if t==8: s(); return
    if t==9:
        et,n=struct.unpack('<IQ',f.read(12))
        for _ in range(n): skip(et)
        return
    f.read(sz[t])
align=32
for _ in range(nkv):
    k=s(); t,=struct.unpack('<I',f.read(4))
    if k==b'general.alignment': align,=struct.unpack('<I',f.read(4))
    else: skip(t)
T={}
for _ in range(nt):
    name=s().decode(); nd,=struct.unpack('<I',f.read(4)); dims=struct.unpack('<%dQ'%nd,f.read(8*nd)); ty,off=struct.unpack('<IQ',f.read(12))
    T[name]=(dims,ty,off)
base=(f.tell()+align-1)//align*align
BS=210  # block_q6_K bytes per 256 values
def slab(name, rows, kblocks, row0=0, kb0=0):
    dims,ty,off=T[name]; assert ty==14, (name,ty)
    K=dims[0]; nb=K//256
    out=bytearray()
    for r in range(row0,row0+rows):
        f.seek(base+off+(r*nb+kb0)*BS); out+=f.read(kblocks*BS)
    return out
jobs=[  # (file, tensor, rows, K) : per-GPU verify shapes
 ('gate_8704x5120.q6k','blk.0.ffn_gate.weight',8704,5120,0,0),
 ('down_5120x8704.q6k','blk.0.ffn_down.weight',5120,8704,0,0),
]
for fn,name,rows,K,r0,k0 in jobs:
    print(name, T[name][0])
    open(fn,'wb').write(slab(name,rows,K//256,r0,k0))
print({n:v[0] for n,v in T.items() if n.startswith('blk.3.') or n.startswith('blk.0.')})
