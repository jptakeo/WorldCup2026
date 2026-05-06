import subprocess

if __name__ == "__main__":
    print("\n=== TREINANDO MODELO PARA 2026 ===\n")
    subprocess.run(["python", "-m", "simulations.train_2026"], check=True)

    print("\n=== SIMULANDO COPA 2026 ===\n")
    subprocess.run(["python", "-m", "simulations.sim_2026"], check=True)