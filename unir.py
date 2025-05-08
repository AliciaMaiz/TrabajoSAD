import pandas as pd
import os

# Muestra archivos disponibles
print("Archivos en carpeta:", os.listdir())

# Cargar los CSV
df1 = pd.read_csv('portugal.csv')
df2 = pd.read_csv('spain.csv')

# Mostrar si tienen datos
print("Filas en df1:", df1.shape[0])
print("Filas en df2:", df2.shape[0])

# Unificación si ambos tienen datos
if not df1.empty and not df2.empty:
    # Asegurar que las columnas coincidan
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df2 = df2[df1.columns]

    # Combinar
    df_combinado = pd.concat([df1, df2], ignore_index=True)
    df_combinado.to_csv('spain-port.csv', index=False)
    print("Combinación completada. Total filas:", df_combinado.shape[0])
else:
    print("Uno de los archivos no tiene datos.")
