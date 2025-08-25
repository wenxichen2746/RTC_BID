import os, math, time, pathlib, itertools, json, multiprocessing as mp
import numpy as np, pandas as pd
from datetime import timedelta
import pynvml

from cfg_eval_flow import (
    EvalConfig, NaiveMethodConfig, RealtimeMethodConfig, BIDMethodConfig,
    CFGMethodConfig, CFGCOS_MethodConfig, main as _main
)
'''
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run src/cfg_eval_flow_batched.py \
  --run_path ./logs-bc/LL_cfg_a4o1_0804 \
  --output-dir ./logs-eval-cfg/08_24_LL \
  --level-paths worlds/l/hard_lunar_lander.json
'''
def _gpu_mem():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0]))
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.total, info.used, info.free

def _fmt(s): return str(timedelta(seconds=int(s)))

def _build_methods(cfg_coef):
    m=[]
    for w_a in cfg_coef:
        m.append((f"cfg_BF_cos:wa{w_a}", {"kind":"CFGCOS", "args":{"w_a":w_a}}))
    m.append(("naive_ca", {"kind":"NAIVE","args":{"mask_action":False}}))
    m.append(("naive_un", {"kind":"NAIVE","args":{"mask_action":True}}))
    m.append(("RTC_un",   {"kind":"RTC","args":{"mask_action":True,"prefix_attention_schedule":"exp","max_guidance_weight":5.0}}))
    m.append(("RTC_hard_un", {"kind":"RTC","args":{"mask_action":True,"prefix_attention_schedule":"zeros","max_guidance_weight":5.0}}))
    m.append(("BID_un",   {"kind":"BID","args":{"mask_action":True,"n_samples":16,"bid_k":None}}))
    for w in cfg_coef+[-1]:
        m.append((f"cfg_BF:wa{w}", {"kind":"CFG","args":{"w_1":0.0,"w_2":0.0,"w_3":1-w,"w_4":w}}))
    for w in cfg_coef:
        m.append((f"cfg_BF:wo{w}", {"kind":"CFG","args":{"w_1":0.0,"w_2":1-w,"w_3":0.0,"w_4":w}}))
    for w_o in cfg_coef:
        w_a=1; w_nn=1-w_o-w_a
        m.append((f"cfg_BI:wo{w_o}", {"kind":"CFG","args":{"w_1":w_nn,"w_2":w_a,"w_3":w_o,"w_4":0.0}}))
    for w_a in cfg_coef:
        w_o=1; w_nn=1-w_o-w_a
        m.append((f"cfg_BI:wa{w_a}", {"kind":"CFG","args":{"w_1":w_nn,"w_2":w_a,"w_3":w_o,"w_4":0.0}}))
    return m

def _make_cfg(base_cfg: EvalConfig, method_spec, inference_delay, execute_horizon):
    name, spec = method_spec
    if spec["kind"]=="NAIVE":
        method = NaiveMethodConfig(**spec["args"])
    elif spec["kind"]=="RTC":
        method = RealtimeMethodConfig(**spec["args"])
    elif spec["kind"]=="BID":
        method = BIDMethodConfig(**spec["args"])
    elif spec["kind"]=="CFG":
        method = CFGMethodConfig(**spec["args"])
    elif spec["kind"]=="CFGCOS":
        method = CFGCOS_MethodConfig(**spec["args"])
    else:
        raise ValueError
    return name, dataclasses.replace(base_cfg, inference_delay=inference_delay, execute_horizon=execute_horizon, method=method)

def _worker_env(mem_fraction: float | None):
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR","platform")
    if mem_fraction is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(mem_fraction)
    return env

def _run_single_method(run_args):
    os.environ.update(_worker_env(run_args["mem_fraction"]))
    args = run_args["args"]
    run_path=args["run_path"]; level_paths=args["level_paths"]; output_dir=args["output_dir"]
    cfg=args["cfg"]; method_name=args["method_name"]; vel_target=args["vel_target"]; noisestd=args["noisestd"]
    execute_horizon=args["execute_horizon"]; seed=args["seed"]
    method_tag=method_name.replace(":","_").replace("/","_")
    outdir = pathlib.Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    m_out = outdir / f"results_{method_tag}__H{execute_horizon}__V{vel_target:.2f}__N{noisestd:.2f}.csv"
    c_out = outdir / f"cosine_{method_tag}__H{execute_horizon}__V{vel_target:.2f}__N{noisestd:.2f}.csv"
    os.environ["FM_METHOD_SINGLE"]=json.dumps({"name":method_name,"vel_target":vel_target,"noisestd":noisestd,"execute_horizon":execute_horizon,"seed":seed,"m_out":str(m_out),"c_out":str(c_out)})
    _main(run_path=run_path, config=cfg, level_paths=tuple(level_paths), seed=seed, output_dir=str(outdir))
    return str(m_out), str(c_out)

def _probe_mem_delta(run_args, warm_cfg: EvalConfig):
    t0_total, u0, f0 = _gpu_mem()
    args = run_args["args"].copy()
    args["cfg"]=dataclasses.replace(warm_cfg, num_evals=64, num_flow_steps=2)
    args["noisestd"]=0.0
    args["execute_horizon"]=1
    args["vel_target"]=0.0
    try:
        _run_single_method({"args":args,"mem_fraction":None})
    except Exception:
        pass
    t1_total, u1, f1 = _gpu_mem()
    return max(256*1024*1024, int(u1-u0))

def run_batched(
    run_path: str,
    level_paths: list[str],
    output_dir: str,
    base_cfg: EvalConfig,
    seed: int = 0,
    inference_delay: int = 1,
    horizons: list[int] = [1,8,3,5],
    vel_list: list[float] = [0.1,0.4,0.7,1.0,1.3],
    noise_list: list[float] = [0.00,0.1,0.2,0.4],
    cfg_coef = list(range(-1,5)),
    reserve_gb: float = 2.0
):
    methods = _build_methods(cfg_coef)
    tasks=[]
    for H in [3,5]:
        tasks.append({"execute_horizon":H,"vel_target":0.0,"noisestd":0.1,"label":"extra_horizon"})
    for H in [1,8]:
        for v in vel_list: tasks.append({"execute_horizon":H,"vel_target":v,"noisestd":0.1,"label":f"vel_target={v:.2f}"})
        for n in noise_list: tasks.append({"execute_horizon":H,"vel_target":0.0,"noisestd":n,"label":f"noise_std={n:.2f}"})

    all_jobs=[]
    for t in tasks:
        for m in methods:
            name, cfg = _make_cfg(base_cfg, m, inference_delay, t["execute_horizon"])
            all_jobs.append({"method":m, "method_name":name, "cfg":cfg, "execute_horizon":t["execute_horizon"], "vel_target":t["vel_target"], "noisestd":t["noisestd"]})

    outdir = pathlib.Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)

    warm_any = next(j for j in all_jobs if True)
    probe_args={"args":{"run_path":run_path,"level_paths":level_paths,"output_dir":output_dir,"cfg":warm_any["cfg"],"method_name":warm_any["method_name"],"vel_target":0.0,"noisestd":0.0,"execute_horizon":1,"seed":seed}}
    per_job = _probe_mem_delta(probe_args, base_cfg)
    total, used, free = _gpu_mem()
    free_eff = max(0, free - int(reserve_gb*1024**3))
    max_parallel = max(1, min(8, free_eff // per_job))  # hard cap 8
    mem_fraction = None
    print(f"Estimated per-job GPU memory usage: {per_job/1024**2:.1f} MiB")
    print(f"GPU memory free total: {free/1024**3:.1f} GiB, used: {used/1024**3:.1f} GiB,{max_parallel=}")
    pool = mp.Pool(processes=max_parallel, maxtasksperchild=1)
    t_all0=time.time()
    futs=[]
    for j in all_jobs:
        run_args={"args":{"run_path":run_path,"level_paths":level_paths,"output_dir":output_dir,"cfg":j["cfg"],"method_name":j["method_name"],"vel_target":j["vel_target"],"noisestd":j["noisestd"],"execute_horizon":j["execute_horizon"],"seed":seed},"mem_fraction":mem_fraction}
        futs.append(pool.apply_async(_run_single_method,(run_args,)))
    pool.close()
    done=0; n=len(futs); durations=[]
    while futs:
        f=futs.pop(0)
        t0=time.time()
        m_out, c_out = f.get()
        dt=time.time()-t0
        durations.append(dt)
        done+=1
        avg=sum(durations)/len(durations)
        rem=avg*(n-done)
        print(f"[{done}/{n}] last={_fmt(dt)} avg={_fmt(avg)} ETA={_fmt(rem)}")
    pool.join()

    res_files = sorted(outdir.glob("results_*.csv"))
    if res_files:
        dfs=[pd.read_csv(p) for p in res_files]
        pd.concat(dfs, ignore_index=True).to_csv(outdir/"results.csv", index=False)
    cos_files = sorted(outdir.glob("cosine_*.csv"))
    if cos_files:
        dfs=[pd.read_csv(p) for p in cos_files]
        pd.concat(dfs, ignore_index=True).to_csv(outdir/"cosine_analysis.csv", index=False)

if __name__=="__main__":
    import dataclasses
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--run_path", required=True)
    p.add_argument("--level-paths", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reserve-gb", type=float, default=2.0)
    a=p.parse_args()
    base=EvalConfig()
    run_batched(a.run_path, a.level_paths, a.output_dir, base, seed=a.seed, reserve_gb=a.reserve_gb)
