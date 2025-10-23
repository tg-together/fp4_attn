import torch







def get_R(QH,KH):

    def batch_sqrtm(H):
       
        eigvals, eigvecs = torch.linalg.eigh(H)  
        sqrtH = eigvecs @ torch.diag_embed(eigvals.clamp(min=0).sqrt()) @ eigvecs.transpose(-2, -1)
        return sqrtH


    KH = KH / KH.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    QH = QH / QH.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)


    # Square roots
    sKH = batch_sqrtm(KH)   
    sQH = batch_sqrtm(QH)  

    # Batched SVD
    U, S, Vh = torch.linalg.svd(sKH @ sQH)   

    # Build R and its inverse per head
    R = sQH @ Vh.transpose(-2, -1) @ torch.diag_embed(S.rsqrt())
    invR = torch.linalg.inv(R)

    return R, invR




def incoherence_processing( QH, KH):

    reg_scale=1e-2

    QH.div_(QH.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))

    QH.diagonal(dim1=-2, dim2=-1).add_(reg_scale)

    KH.div_(KH.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))
    KH.diagonal(dim1=-2, dim2=-1).add_(reg_scale)

    R, invR=get_R(QH,KH)

    return R, invR

# model_strs = ["meta-llama/Llama-3.2-3B-Instruct", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen/Qwen3-4B-Thinking-2507","meta-llama/Llama-3.3-70B-Instruct",  "Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "meta-llama/Llama-3.1-70B"]
model_strs = ["meta-llama/Llama-3.1-8B-Instruct"]
for model_str in model_strs:
    mag_reduce={}
    model_short = model_str.split('/')[-1] if '/' in model_str else model_str
    hessian_folder = f"dumps/{model_short}_wikitext2"
    qk_hessian_file = f"{hessian_folder}/qk_hessians.pt"
    qk_hessians=torch.load(qk_hessian_file)
    
    for key in qk_hessians.keys():
        print(key)
        q_hessian=qk_hessians[key]['q_hessian'].to(torch.float32)
        k_hessian=qk_hessians[key]['k_hessian'].to(torch.float32)
        R, invR=incoherence_processing(q_hessian, k_hessian)
        mag_reduce[key]={'R':R, 'invR':invR}

    torch.save(mag_reduce, f"{hessian_folder}/mag_reduce.pt")
