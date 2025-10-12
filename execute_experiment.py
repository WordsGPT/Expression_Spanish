"""
Usage:
    python3 execute_experiment.py <EXPERIMENT NAME> <EXPERIMENT_PATH>

    EXPERIMENT_NAME:
         * <experiment_name>: process specific experiment.
         * "all": process all experiments in the batches folder.
         * "remain": check and download batches still in tracking.
         * "status": show batches still in tracking.
         * "failed": re-execute all previously failed experiments.
    
    EXPERIMENT_PATH:
         * Path to the experiment folder containing config.yaml, batches/, results/, etc.
"""

import os
import sys
import time
from datetime import datetime

from utils import load_config, openai_login, get_experiment_paths, ensure_experiment_directories

import pandas as pd
import hashlib

def get_experiment_prefixes_from_batches(experiment_path):
    """Devuelve una lista de prefijos únicos de experimentos en la carpeta batches."""
    paths = get_experiment_paths(experiment_path)
    prefixes = set()
    if os.path.exists(paths['batches']):
        for fname in os.listdir(paths['batches']):
            if "_prompt" in fname:
                prefix = fname.split("_prompt")[0] + "_prompt"
                prefixes.add(prefix)
    return list(prefixes)

def get_batches_for_experiment(prefix, experiment_path):
    """Devuelve la lista de batches que empiezan por el prefijo dado."""
    paths = get_experiment_paths(experiment_path)
    if os.path.exists(paths['batches']):
        return [f for f in os.listdir(paths['batches']) if f.startswith(prefix)]
    return []

def create_batch_tracking_file(experiment_path, file_path=None):
    """Crea un archivo Excel para trackear el estado de los batches."""
    paths = get_experiment_paths(experiment_path)
    if file_path is None:
        file_path = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            'experiment_name', 'batch_file', 'batch_id', 'status', 'timestamp', 'file_hash'
        ])
        df.to_excel(file_path, index=False)
        print(f"Archivo de tracking creado: {file_path}")
    return file_path

def load_batch_tracking(experiment_path, file_path=None):
    """Carga el archivo Excel de tracking de batches."""
    if file_path is None:
        paths = get_experiment_paths(experiment_path)
        file_path = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error cargando archivo de tracking: {e}")
            return pd.DataFrame(columns=[
                'experiment_name', 'batch_file', 'batch_id', 'status', 'timestamp', 'file_hash'
            ])
    else:
        return pd.DataFrame(columns=[
            'experiment_name', 'batch_file', 'batch_id', 'status', 'timestamp', 'file_hash'
        ])

def save_batch_tracking(df, experiment_path, file_path=None):
    """Guarda el DataFrame de tracking en el archivo Excel."""
    if file_path is None:
        paths = get_experiment_paths(experiment_path)
        file_path = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    try:
        df.to_excel(file_path, index=False)
    except Exception as e:
        print(f"Error guardando archivo de tracking: {e}")

def get_file_hash(file_path, experiment_path):
    """Calcula el hash MD5 de un archivo para detectar cambios."""
    paths = get_experiment_paths(experiment_path)
    full_path = os.path.join(paths['batches'], file_path) if not os.path.exists(file_path) else file_path
    with open(full_path, 'rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def add_batch_to_tracking(experiment_name, batch_file, batch_id, experiment_path, file_path=None):
    """Añade un nuevo batch al archivo de tracking."""
    df = load_batch_tracking(experiment_path, file_path)
    file_hash = get_file_hash(batch_file, experiment_path)
    new_row = pd.DataFrame({
        'experiment_name': [experiment_name],
        'batch_file': [batch_file],
        'batch_id': [batch_id],
        'status': ['submitted'],
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'file_hash': [file_hash]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    save_batch_tracking(df, experiment_path, file_path)

def update_batch_status(batch_id, status, experiment_path, file_path=None):
    """Actualiza el estado de un batch en el archivo de tracking."""
    df = load_batch_tracking(experiment_path, file_path)
    df.loc[df['batch_id'] == batch_id, 'status'] = status
    save_batch_tracking(df, experiment_path, file_path)

def remove_batch_from_tracking(batch_id, experiment_path, file_path=None):
    """Elimina un batch del archivo de tracking."""
    df = load_batch_tracking(experiment_path, file_path)
    df = df[df['batch_id'] != batch_id]
    save_batch_tracking(df, experiment_path, file_path)
    print(f"Batch {batch_id} eliminado del tracking.")

def cleanup_empty_tracking_file(experiment_path, file_path=None):
    """Elimina el archivo de tracking si está vacío."""
    if file_path is None:
        paths = get_experiment_paths(experiment_path)
        file_path = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    if os.path.exists(file_path):
        df = load_batch_tracking(experiment_path, file_path)
        if len(df) == 0:
            os.remove(file_path)
            print(f"Archivo de tracking {file_path} eliminado (no hay batches pendientes).")
            return True
        else:
            print(f"Quedan {len(df)} batches en tracking.")
            return False
    return True

def get_pending_batches_for_experiment(experiment_name, experiment_path, file_path=None):
    """Devuelve los batches pendientes para un experimento específico."""
    df = load_batch_tracking(experiment_path, file_path)
    # Filtrar por experimento y que no estén completados o descargados
    pending = df[
        (df['experiment_name'] == experiment_name) & 
        (~df['status'].isin(['completed', 'downloaded', 'failed']))
    ]
    return pending

def get_batches_to_send(experiment_name, batch_files, experiment_path, file_path=None):
    """Determina qué batches necesitan ser enviados y cuáles ya están en tracking."""
    df = load_batch_tracking(experiment_path, file_path)
    existing_batches = df[df['experiment_name'] == experiment_name]
    
    batches_to_send = []
    for batch_file in batch_files:
        # Verificar si el batch ya existe en tracking
        file_hash = get_file_hash(batch_file, experiment_path)
        existing = existing_batches[
            (existing_batches['batch_file'] == batch_file) & 
            (existing_batches['file_hash'] == file_hash)
        ]
        
        if existing.empty:
            # El batch no existe o ha cambiado, necesita ser enviado
            batches_to_send.append(batch_file)
        else:
            print(f"Batch {batch_file} ya está en tracking, saltando envío.")
    
    return batches_to_send

def save_failed_experiment_info(batch_id, experiment_name, batch_file, status, experiment_path):
    """Guarda el nombre del experimento en la lista de experimentos fallidos."""
    # Agregar el nombre del experimento al archivo de experimentos fallidos para reejecutar
    add_failed_experiment_to_list(experiment_name, experiment_path)

def add_failed_experiment_to_list(experiment_name, experiment_path):
    """Agrega el nombre del experimento a la lista de experimentos fallidos para reejecutar."""
    failed_experiments_dir = os.path.join(experiment_path, "failed_experiments")
    
    # Crear directorio si no existe
    if not os.path.exists(failed_experiments_dir):
        os.makedirs(failed_experiments_dir)
    
    failed_list_file = os.path.join(failed_experiments_dir, "failed_execute_exp.txt")
    
    try:
        # Leer experimentos ya fallidos para evitar duplicados
        existing_failed = set()
        if os.path.exists(failed_list_file):
            with open(failed_list_file, 'r', encoding='utf-8') as f:
                existing_failed = set(line.strip() for line in f if line.strip())
        
        # Agregar el nuevo experimento fallido si no está ya en la lista
        if experiment_name not in existing_failed:
            with open(failed_list_file, 'a', encoding='utf-8') as f:
                f.write(f"{experiment_name}\n")
            print(f"Experimento '{experiment_name}' agregado a la lista de experimentos fallidos.")
        else:
            print(f"Experimento '{experiment_name}' ya estaba en la lista de experimentos fallidos.")
            
    except Exception as e:
        print(f"Error agregando experimento a la lista de fallidos: {e}")

def get_failed_experiments_list(experiment_path):
    """Obtiene la lista de experimentos fallidos para reejecutar."""
    failed_list_file = os.path.join(experiment_path, "failed_experiments", "failed_execute_exp.txt")
    
    if not os.path.exists(failed_list_file):
        return []
    
    try:
        with open(failed_list_file, 'r', encoding='utf-8') as f:
            failed_experiments = [line.strip() for line in f if line.strip()]
        return failed_experiments
    except Exception as e:
        print(f"Error leyendo lista de experimentos fallidos: {e}")
        return []

def remove_experiment_from_failed_list(experiment_name, experiment_path):
    """Elimina un experimento de la lista de fallidos cuando se completa exitosamente."""
    failed_list_file = os.path.join(experiment_path, "failed_experiments", "failed_execute_exp.txt")
    
    if not os.path.exists(failed_list_file):
        return
    
    try:
        # Leer todos los experimentos
        with open(failed_list_file, 'r', encoding='utf-8') as f:
            experiments = [line.strip() for line in f if line.strip()]
        
        # Filtrar el experimento completado
        updated_experiments = [exp for exp in experiments if exp != experiment_name]
        
        # Reescribir el archivo
        if updated_experiments:
            with open(failed_list_file, 'w', encoding='utf-8') as f:
                for exp in updated_experiments:
                    f.write(f"{exp}\n")
        else:
            # Si no quedan experimentos fallidos, eliminar el archivo
            os.remove(failed_list_file)
            print(f"Lista de experimentos fallidos eliminada (no quedan experimentos pendientes).")
            
        print(f"Experimento '{experiment_name}' eliminado de la lista de experimentos fallidos.")
        
    except Exception as e:
        print(f"Error eliminando experimento de la lista de fallidos: {e}")

def check_and_download_pending_batches(experiment_path):
    """Comprueba el estado de todos los batches pendientes y descarga los completados."""
    paths = get_experiment_paths(experiment_path)
    tracking_file = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    if not os.path.exists(tracking_file):
        print("No hay archivo de tracking de batches.")
        return
    
    df = load_batch_tracking(experiment_path, tracking_file)
    
    if df.empty:
        print("No hay batches en tracking.")
        return
    
    print("\n=== Comprobando estado de todos los batches pendientes ===")
    check_and_download_openai_batches(df, experiment_path, tracking_file)
    
    # Limpiar archivo de tracking si está vacío después de las descargas
    cleanup_empty_tracking_file(experiment_path, tracking_file)

def check_and_download_openai_batches(company_batches, experiment_path, tracking_file):
    """Comprueba y descarga batches de OpenAI."""
    try:
        paths = get_experiment_paths(experiment_path)
        client = openai_login()
        date_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        # Filtrar solo batches que no están descargados
        pending_batches = company_batches[
            ~company_batches['status'].isin(['downloaded', 'failed'])
        ]
        
        for _, row in pending_batches.iterrows():
            batch_id = row['batch_id']
            prefix = row['experiment_name']
            batch_file = row['batch_file']
            current_status = row['status']
            
            try:
                # Comprobar estado actual en OpenAI
                batch_info = client.batches.retrieve(batch_id)
                new_status = batch_info.status
                
                print(f"Batch {batch_id} ({batch_file}): {current_status} -> {new_status}")
                
                # Actualizar estado si ha cambiado
                if new_status != current_status:
                    update_batch_status(batch_id, new_status, experiment_path, tracking_file)
                
                # Descargar si está completado
                if new_status == "completed":
                    try:
                        batch_results_id = batch_info.output_file_id
                        result = client.files.content(batch_results_id).content
                        save_path = os.path.join(paths['results'], f"{prefix}_results_{batch_id}_{date_string}.jsonl")
                        
                        with open(save_path, "wb") as file:
                            file.write(result)
                        
                        print(f"Descargado resultado en {save_path}")
                        
                        # Eliminar del tracking después de descargar exitosamente
                        remove_batch_from_tracking(batch_id, experiment_path, tracking_file)
                        
                        # Eliminar de la lista de experimentos fallidos si estaba ahí
                        remove_experiment_from_failed_list(prefix, experiment_path)
                        
                    except Exception as e:
                        print(f"Error descargando resultado batch {batch_id}: {e}")
                
                elif new_status in ["failed", "cancelled"]:
                    print(f"Batch {batch_id} falló o fue cancelado. Estado: {new_status}")
                    # Guardar información del experimento fallido
                    save_failed_experiment_info(batch_id, prefix, batch_file, new_status, experiment_path)
                    # Eliminar del tracking después de guardar la información del fallo
                    remove_batch_from_tracking(batch_id, experiment_path, tracking_file)
                    
            except Exception as e:
                print(f"Error comprobando batch {batch_id}: {e}")
                
    except Exception as e:
        print(f"Error conectando con OpenAI: {e}")

def execute_openai(config_args, experiment_prefixes, experiment_path):
    client = openai_login()
    paths = get_experiment_paths(experiment_path)
    tracking_file = create_batch_tracking_file(experiment_path)
    
    for prefix in experiment_prefixes:
        batch_files = get_batches_for_experiment(prefix, experiment_path)
        
        # Verificar qué batches necesitan ser enviados (evitar duplicados)
        batches_to_send = get_batches_to_send(prefix, batch_files, experiment_path, tracking_file)
        
        if not batches_to_send:
            print(f"Todos los batches para {prefix} ya están en tracking. Use 'remain' para verificar su estado.")
            # Si el experimento tiene batches (aunque ya estén en tracking), 
            # se considera que fue procesado exitosamente, así que eliminarlo de la lista de fallidos
            if batch_files:
                remove_experiment_from_failed_list(prefix, experiment_path)
            continue
        
        # Si llegamos aquí, el experimento se está procesando exitosamente
        experiment_processed_successfully = False
        
        for file_name in batches_to_send:
            try:
                batch_path = os.path.join(paths['batches'], file_name)
                with open(batch_path, "rb") as f:
                    batch_file = client.files.create(
                        file=f, purpose="batch"
                    )
                batch_job = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                print(f"# File Named: {file_name} has been submitted for processing")
                print(f"# Batch Job ID: {batch_job.id}")
                
                # Guardar en tracking
                add_batch_to_tracking(prefix, file_name, batch_job.id, experiment_path, tracking_file)
                experiment_processed_successfully = True
                
            except Exception as e:
                print(f"Error enviando batch {file_name}: {e}")
                # Guardar información del error de envío
                save_failed_experiment_info("ERROR_SENDING", prefix, file_name, f"Error de envío: {str(e)}", experiment_path)
        
        # Si al menos un batch se procesó exitosamente, eliminar de la lista de fallidos
        if experiment_processed_successfully:
            remove_experiment_from_failed_list(prefix, experiment_path)
    
    # Después de enviar todos los batches, comprobar estado cada 120 segundos
    print("\n=== Iniciando comprobación periódica de batches cada 120 segundos ===")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprobando estado de batches...")
        check_and_download_pending_batches(experiment_path)
        
        # Verificar si quedan batches pendientes
        df = load_batch_tracking(experiment_path, tracking_file)
        if df.empty:
            print("No quedan batches pendientes. Finalizando comprobación.")
            break
        
        print("Esperando 120 segundos antes de la próxima comprobación...")
        time.sleep(120)

def check_and_download_pending_batches_loop(experiment_path):
    """Comprueba el estado de todos los batches pendientes cada 120 segundos hasta que no queden."""
    paths = get_experiment_paths(experiment_path)
    tracking_file = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    if not os.path.exists(tracking_file):
        print("No hay archivo de tracking de batches.")
        return
    
    print("\n=== Iniciando comprobación periódica de batches cada 120 segundos ===")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comprobando estado de batches...")
        check_and_download_pending_batches(experiment_path)
        
        # Verificar si quedan batches pendientes
        df = load_batch_tracking(experiment_path, tracking_file)
        if df.empty:
            print("No quedan batches pendientes. Finalizando comprobación.")
            break
        
        print(f"Quedan {len(df)} batches pendientes. Esperando 120 segundos...")
        time.sleep(120)

def show_batch_status(experiment_path, file_path=None):
    """Muestra el estado actual de todos los batches en tracking."""
    if file_path is None:
        paths = get_experiment_paths(experiment_path)
        file_path = os.path.join(paths['results'], "batch_tracking.xlsx")
    
    df = load_batch_tracking(experiment_path, file_path)
    if df.empty:
        print("No hay batches en tracking.")
        return
    
    print("\n=== Estado de Batches ===")
    print(f"{'Experimento':<20} {'Archivo':<30} {'Estado':<12} {'Timestamp':<20}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['experiment_name']:<20} {row['batch_file']:<30} {row['status']:<12} {row['timestamp']:<20}")
    
    # Resumen por estado
    status_counts = df['status'].value_counts()
    print(f"\n=== Resumen ===")
    for status, count in status_counts.items():
        print(f"{status}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python execute_experiment.py <EXPERIMENT_NAME> <EXPERIMENT_PATH>")
        print("  EXPERIMENT_NAME options: <experiment_name>, all, status, remain, failed")
        print("  EXPERIMENT_PATH: Path to experiment folder")
        print("  - failed: ejecuta todos los experimentos que han fallado previamente")
        exit()

    EXPERIMENT_NAME = sys.argv[1]
    EXPERIMENT_PATH = sys.argv[2]
    
    # Ensure experiment directories exist
    ensure_experiment_directories(EXPERIMENT_PATH)

    if EXPERIMENT_NAME.lower() == "status":
        show_batch_status(EXPERIMENT_PATH)
        exit()
    
    # Comando para comprobar y descargar batches pendientes
    if EXPERIMENT_NAME.lower() == "remain":
        print("Comprobando y descargando todos los batches pendientes...")
        check_and_download_pending_batches_loop(EXPERIMENT_PATH)
        exit()

    # Comando para ejecutar experimentos fallidos
    if EXPERIMENT_NAME.lower() == "failed":
        failed_experiments = get_failed_experiments_list(EXPERIMENT_PATH)
        if not failed_experiments:
            print("No hay experimentos fallidos para reejecutar.")
            exit()
        
        print(f"\nEjecutando experimentos fallidos: {failed_experiments}")
        # Verificar que todos los experimentos tienen batches disponibles
        available_experiments = []
        for exp_name in failed_experiments:
            batch_files = get_batches_for_experiment(exp_name, EXPERIMENT_PATH)
            if batch_files:
                available_experiments.append(exp_name)
            else:
                print(f"Advertencia: No se encontraron batches para el experimento fallido '{exp_name}'")
        
        if not available_experiments:
            print("No se encontraron batches para ningún experimento fallido.")
            exit()
        
        config_args = load_config(
            config_type="experiments",
            name=available_experiments[0],
            experiment_path=EXPERIMENT_PATH
        )
        execute_openai(config_args, available_experiments, EXPERIMENT_PATH)
        exit()

    # Ejecución de experimentos
    if EXPERIMENT_NAME.lower() == "all":
        prefixes = get_experiment_prefixes_from_batches(EXPERIMENT_PATH)
        if not prefixes:
            print("No se encontraron experimentos en la carpeta batches.")
            exit()
        
        print(f"\nEjecutando todos los experimentos con OpenAI: {prefixes}")
        config_args = load_config(
            config_type="experiments",
            name=prefixes[0],
            experiment_path=EXPERIMENT_PATH
        )
        execute_openai(config_args, prefixes, EXPERIMENT_PATH)
    else:
        # Ejecuta solo el experimento indicado
        # Verificar si existe el batch correspondiente
        batch_files = get_batches_for_experiment(EXPERIMENT_NAME, EXPERIMENT_PATH)
        if not batch_files:
            print(f"No se encontraron batches para el experimento '{EXPERIMENT_NAME}' en la carpeta batches.")
            # Registrar como experimento fallido
            save_failed_experiment_info("NO_BATCH_FILES", EXPERIMENT_NAME, "N/A", "No se encontraron archivos de batch", EXPERIMENT_PATH)
            exit()
        
        config_args = load_config(
            config_type="experiments",
            name=EXPERIMENT_NAME,
            experiment_path=EXPERIMENT_PATH
        )
        
        print(f"Ejecutando experimento {EXPERIMENT_NAME} con OpenAI")
        prefixes = [EXPERIMENT_NAME]
        execute_openai(config_args, prefixes, EXPERIMENT_PATH)