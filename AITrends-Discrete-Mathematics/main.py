import numpy as np
import pandas as pd
import os
import random
import time
from colorama import init, Fore, Back, Style
# Vector de probabilidad inicial
VECTOR_INICIAL = {
    'OpenAI': 0.3,
    'GoogleAI': 0.25,
    'Anthropic': 0.15,
    'Cohere': 0.2,
    'HuggingFace': 0.1
}

# Inicializar colorama
init(autoreset=True)

def limpiar_consola():
    os.system('cls')
    print(Fore.CYAN + "=" * 60)
    print(Fore.YELLOW + "ANÁLISIS DE PLATAFORMAS DE INTELIGENCIA ARTIFICIAL".center(60))
    print(Fore.CYAN + "=" * 60 + "\n")

def obtener_datos_ia_plataformas():
    import csv
    
    # Lista de plataformas (debe coincidir con VECTOR_INICIAL)
    plataformas = list(VECTOR_INICIAL.keys())
    datos = []
    
    try:
        # Mostrar la ruta del archivo para depuración
        ruta_archivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos_ia_realistas.csv')
        print(f"Intentando leer archivo: {ruta_archivo}")
        
        # Leer el archivo CSV
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            lineas = archivo.readlines()
            
            # Saltar la primera línea (encabezado)
            for i, linea in enumerate(lineas[1:], 1):
                try:
                    # Dividir la línea por comas, ignorando el ID del usuario
                    partes = linea.strip().split(',')
                    if len(partes) < 2:  # Si no hay suficientes partes, saltar
                        continue
                        
                    # Tomar todas las partes después del ID del usuario
                    secuencia = [p.strip() for p in partes[1:] if p.strip()]
                    
                    # Validar que todas las plataformas en la secuencia sean válidas
                    if all(plataforma in plataformas for plataforma in secuencia):
                        if len(secuencia) > 1:  # Solo agregar si hay al menos 2 elementos para transición
                            datos.append(secuencia)
                    else:
                        plataformas_invalidas = [p for p in secuencia if p not in plataformas]
                        print(f"Advertencia: Línea {i+1} contiene plataformas no reconocidas: {plataformas_invalidas}")
                except Exception as e:
                    print(f"Error al procesar la línea {i+1}: {str(e)}")
        
        print(f"Se cargaron {len(datos)} secuencias válidas del archivo.")
        
        # Si no hay datos, usar datos de muestra
        if not datos:
            print("No se encontraron datos válidos en el archivo")
            
            
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo en {ruta_archivo}")
        
    return datos, plataformas


def calcular_matriz_transicion(datos, estados):
    estado_a_indice = {estado: i for i, estado in enumerate(estados)}
    n = len(estados)
    
    # Inicializar matriz de conteos con suavizado de Laplace (suma 1 a cada celda)
    # Esto evita divisiones por cero y da una distribución uniforme inicial
    conteos = np.ones((n, n))
    
    # Contar transiciones
    for secuencia in datos:
        for i in range(len(secuencia)-1):
            actual = secuencia[i]
            siguiente = secuencia[i+1]
            conteos[estado_a_indice[actual]][estado_a_indice[siguiente]] += 1
    
    # Normalizar cada fila para que sume 1
    sumas_filas = conteos.sum(axis=1, keepdims=True)
    # Reemplazar ceros por 1 para evitar divisiones por cero
    sumas_filas[sumas_filas == 0] = 1
    matriz = conteos / sumas_filas
    
    # Verificación de que cada fila suma 1 (con cierta tolerancia por errores de punto flotante)
    for i in range(n):
        suma_fila = matriz[i].sum()
        if not np.isclose(suma_fila, 1.0, atol=1e-10):
            print(f"Advertencia: La fila {i} suma {suma_fila}, normalizando...")
            matriz[i] = matriz[i] / suma_fila
    
    return matriz

def calcular_matriz_orden_n(matriz, n):
    limpiar_consola()
    """
    Calcula la matriz de transición de orden n elevando la matriz de transición a la n-ésima potencia.
    
    Args:
        matriz: Matriz de transición estocástica (filas suman 1)
        n: Número de pasos
        
    Returns:
        Matriz de transición después de n pasos
    """
    # Verificar que la matriz es cuadrada
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz de transición debe ser cuadrada")
        
    # Verificar que cada fila suma 1 (con cierta tolerancia)
    for i in range(matriz.shape[0]):
        suma_fila = np.sum(matriz[i])
        if not np.isclose(suma_fila, 1.0, atol=1e-8):
            print(f"Advertencia: La fila {i} suma {suma_fila}, normalizando...")
            matriz[i] = matriz[i] / suma_fila
    
    # Calcular la matriz elevada a la n-ésima potencia
    return np.linalg.matrix_power(matriz, n)

def calcular_distribucion_n_pasos(vector_inicial, matriz_transicion, n):
    limpiar_consola()
    """
    Calcula la distribución de probabilidad después de n pasos.
    
    Args:
        vector_inicial: Vector de probabilidad inicial (debe sumar 1)
        matriz_transicion: Matriz de transición
        n: Número de pasos
        
    Returns:
        Vector de probabilidad después de n pasos
    """
    # Verificar que el vector inicial suma 1
    if not np.isclose(np.sum(vector_inicial), 1.0, atol=1e-8):
        print("Advertencia: El vector inicial no suma 1, normalizando...")
        vector_inicial = vector_inicial / np.sum(vector_inicial)
    
    # Calcular la matriz de transición en el paso n
    matriz_n = calcular_matriz_orden_n(matriz_transicion, n)
    
    # Multiplicar el vector inicial por la matriz de transición elevada a n
    distribucion = np.dot(vector_inicial, matriz_n)
    
    # Asegurar que la distribución suma 1 (por posibles errores de redondeo)
    distribucion = distribucion / np.sum(distribucion)
    
    return distribucion

# Obtener datos y calcular la matriz de transición
datos, plataformas_ia = obtener_datos_ia_plataformas()
matriz = calcular_matriz_transicion(datos, plataformas_ia)

# Convertir el vector inicial a un array de numpy
vector_inicial = np.array(list(VECTOR_INICIAL.values()))
# Normalizar el vector inicial
vector_inicial = vector_inicial / np.sum(vector_inicial)

def mostrar_datos_navegacion():
    limpiar_consola()
    print("\n" + Fore.CYAN + "=" * 60)
    print(Fore.YELLOW + "CARGANDO DATOS DE NAVEGACIÓN...".center(60))
    print(Fore.CYAN + "=" * 60 + "\n")
    
    # Obtener los datos actualizados
    datos_actuales, _ = obtener_datos_ia_plataformas()
    
    mostrar_encabezado("📊 DATOS DE NAVEGACIÓN ENTRE PLATAFORMAS DE IA")
    
    print(Fore.CYAN + "Plataformas disponibles:" + Fore.YELLOW + f" {', '.join(plataformas_ia)}" + "\n")
    
    # Colores para las diferentes plataformas
    colores = {
        'OpenAI': Fore.BLUE,
        'GoogleAI': Fore.RED,
        'Anthropic': Fore.GREEN,
        'Cohere': Fore.MAGENTA,
        'HuggingFace': Fore.YELLOW
    }
    
    # Mostrar las primeras 5 secuencias o menos si hay menos de 5
    num_a_mostrar = min(5, len(datos_actuales))
    
    if num_a_mostrar == 0:
        print(Fore.RED + "No se encontraron secuencias de navegación para mostrar.")
        mostrar_pie()
        input("\nPresiona Enter para continuar...")
        return
    
    print(Fore.WHITE + f"🔍 Mostrando {num_a_mostrar} de {len(datos_actuales)} secuencias de usuarios:")
    print(Fore.CYAN + "─" * 60)
    
    for i in range(num_a_mostrar):
        secuencia = datos_actuales[i]
        secuencia_coloreada = []
        
        # Crear flechas de transición
        transiciones = []
        for j in range(len(secuencia) - 1):
            plataforma_actual = secuencia[j]
            plataforma_siguiente = secuencia[j + 1]
            color_actual = colores.get(plataforma_actual, Fore.WHITE)
            color_siguiente = colores.get(plataforma_siguiente, Fore.WHITE)
            
            transicion = f"{color_actual}{plataforma_actual}{Style.RESET_ALL} → {color_siguiente}{plataforma_siguiente}"
            transiciones.append(transicion)
        
        # Mostrar la secuencia completa
        print(f"\n👤 {Fore.CYAN}Usuario {i+1}:{Style.RESET_ALL}")
        print(f"   {' → '.join([f'{colores.get(p, Fore.WHITE)}{p}{Style.RESET_ALL}' for p in secuencia])}")
        
        # Mostrar transiciones individuales
        print(f"\n   {Fore.YELLOW}Transiciones:{Style.RESET_ALL}")
        for j, trans in enumerate(transiciones, 1):
            print(f"   {j}. {trans}")
    
    if len(datos_actuales) > 5:
        print(f"\n{Fore.CYAN}... y {len(datos_actuales) - 5} secuencias más ...")
    
    print("\n" + Fore.YELLOW + "💡 Información:" + Fore.WHITE + " Estas secuencias muestran cómo los usuarios ")
    print("transicionan entre diferentes plataformas de IA. Cada flecha (→) representa")
    print("un cambio de plataforma realizado por el usuario.")
    
    mostrar_pie()
    

def construir_matriz_transicion():
    limpiar_consola()
    mostrar_encabezado("📈 MATRIZ DE TRANSICIÓN ENTRE PLATAFORMAS DE IA")
    
    print(Fore.WHITE + "ℹ  Esta matriz muestra las probabilidades de transición entre plataformas:")
    print(Fore.YELLOW + "🔍 Valores más altos indican transiciones más probables entre plataformas.\n")
    
    # Crear un DataFrame con formato mejorado
    df = pd.DataFrame(
        np.round(matriz, 2),  # Redondear a 2 decimales
        index=plataformas_ia,
        columns=plataformas_ia
    )
    
    # Configuración para mostrar el DataFrame
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 10)
    
    # Función para aplicar formato de color al DataFrame
    def colorize(val):
        if val > 0.6:
            return f'background-color: #4CAF50; color: white; font-weight: bold;'
        elif val > 0.3:
            return 'background-color: #8BC34A; color: black;'
        elif val > 0.1:
            return 'background-color: #FFEB3B; color: black;'
        elif val > 0:
            return 'background-color: #FF9800; color: white;'
        else:
            return ''
    
    # Mostrar el DataFrame con formato de color
    try:
        from IPython.display import display, HTML
        display(HTML(df.style.applymap(colorize).to_html()))
    except:
        # Si no se puede mostrar con colores, mostrar sin formato
        print(df.to_string())
    
    print("\n" + Fore.CYAN + "📝 Interpretación:" + Fore.WHITE + " Por ejemplo, un valor de " + 
          Fore.YELLOW + "0.50" + Fore.WHITE + " en la fila 'OpenAI' y columna 'GoogleAI'")
    print("indica que hay un " + Fore.YELLOW + "50%" + Fore.WHITE + " de probabilidad de que un usuario que usa " + 
          Fore.BLUE + "OpenAI" + Fore.WHITE + " luego use " + Fore.RED + "Google AI" + Fore.WHITE + ".")
    
    mostrar_pie()
    

def pronosticar_probabilidad():
    limpiar_consola()
    mostrar_encabezado("🔮 PRONÓSTICO PARA N PASOS")
    
    print(Fore.CYAN + "Ingresa el número de pasos (n) para el pronóstico:" + Fore.WHITE)
    n = int(input("➤ "))
    
    print("\n" + Fore.YELLOW + "⏳ Calculando matriz de transición..." + Style.RESET_ALL)
    time.sleep(1)
    
    matriz_n = calcular_matriz_orden_n(matriz, n)
    df = pd.DataFrame(matriz_n, index=plataformas_ia, columns=plataformas_ia)
    
    # Mostrar la matriz con formato mejorado
    print("\n" + Fore.GREEN + "═" * 85)
    print(Fore.YELLOW + f"📊 MATRIZ DE TRANSICIÓN A {n} PASOS".center(85))
    print(Fore.GREEN + "═" * 85 + Style.RESET_ALL)
    
    # Configurar opciones de visualización de pandas
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.float_format', '{:.2%}'.format):
        print(Fore.CYAN + str(df) + Style.RESET_ALL)
    
    print(Fore.GREEN + "═" * 85 + Style.RESET_ALL)
    
    # Mostrar interpretación
    print("\n" + Fore.YELLOW + "💡 Interpretación:" + Style.RESET_ALL)
    print(Fore.WHITE + "Cada celda muestra la probabilidad de transición después de " + 
          Fore.CYAN + f"{n} pasos" + Fore.WHITE + " desde la")
    print(Fore.WHITE + "plataforma de la fila a la de la columna.")
    
    mostrar_pie()

def pronosticar_largo_plazo():
    limpiar_consola()
    mostrar_encabezado("📈 DISTRIBUCIÓN A LARGO PLAZO")
    
    print(Fore.YELLOW + "⏳ Calculando distribución estacionaria..." + Style.RESET_ALL)
    time.sleep(1)
    
    # Calcular la distribución estacionaria
    valores_propios, vectores_propios = np.linalg.eig(matriz.T)
    indice = np.argmin(np.abs(valores_propios - 1.0))
    vector_estacionario = vectores_propios[:, indice].real
    vector_estacionario = np.abs(vector_estacionario)  # Asegurar valores positivos
    vector_estacionario /= vector_estacionario.sum()  # Normalizar
    
    # Mostrar resultados con formato mejorado
    print("\n" + Fore.GREEN + "═" * 60)
    print(Fore.YELLOW + "📊 DISTRIBUCIÓN ESTACIONARIA".center(60))
    print(Fore.GREEN + "═" * 60 + Style.RESET_ALL)
    
    # Calcular el ancho máximo para la alineación
    max_len = max(len(plataforma) for plataforma in plataformas_ia)
    
    # Mostrar cada plataforma con su probabilidad y barra de progreso
    for plataforma, prob in sorted(zip(plataformas_ia, vector_estacionario), key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 50)  # Barra de progreso
        print(f"{Fore.CYAN}{plataforma.ljust(max_len)} {Fore.WHITE}[{prob:.2%}] {Fore.GREEN}{barra}")
    
    print(Fore.GREEN + "═" * 60 + Style.RESET_ALL)
    
    # Mostrar interpretación
    print("\n" + Fore.YELLOW + "💡 Interpretación:" + Style.RESET_ALL)
    print(Fore.WHITE + "Estas son las probabilidades a largo plazo de que un usuario esté en cada plataforma, ")
    print(Fore.WHITE + "independientemente de la plataforma inicial.")
    
    mostrar_pie()

def generar_recomendaciones():
    limpiar_consola()
    mostrar_encabezado("💡 RECOMENDACIONES DE NAVEGACIÓN")
    
    print(Fore.YELLOW + "⏳ Analizando patrones de navegación..." + Style.RESET_ALL)
    time.sleep(1)
    
    print("\n" + Fore.GREEN + "═" * 80)
    print(Fore.YELLOW + "📋 RECOMENDACIONES POR PLATAFORMA".center(80))
    print(Fore.GREEN + "═" * 80 + Style.RESET_ALL)
    
    # Colores para las diferentes plataformas
    colores = {
        'OpenAI': Fore.BLUE,
        'GoogleAI': Fore.RED,
        'Anthropic': Fore.GREEN,
        'Cohere': Fore.MAGENTA,
        'HuggingFace': Fore.YELLOW
    }
    
    for plataforma in plataformas_ia:
        transiciones = matriz[plataformas_ia.index(plataforma)]
        top_transiciones = sorted(zip(plataformas_ia, transiciones), key=lambda x: x[1], reverse=True)[:2]
        
        # Determinar el color de la plataforma actual
        color_actual = colores.get(plataforma, Fore.WHITE)
        
        print(f"\n{Fore.CYAN}┌{Fore.WHITE} Usuarios de {color_actual}{plataforma}{Style.RESET_ALL}")
        
        for i, (plataforma_destino, prob) in enumerate(top_transiciones, 1):
            color_destino = colores.get(plataforma_destino, Fore.WHITE)
            barra = "▌" * int(prob * 30)  # Barra de progreso más corta
            print(f"{Fore.CYAN}│{Fore.WHITE} {i}. {prob:.1%} → {color_destino}{plataforma_destino}{Style.RESET_ALL} {barra}")
        
        # Mostrar una recomendación basada en las transiciones
        mejor_destino = top_transiciones[0][0]
        color_mejor_destino = colores.get(mejor_destino, Fore.WHITE)
        print(f"{Fore.CYAN}│{Fore.YELLOW}   💡 Recomendación: Considera integrar con {color_mejor_destino}{mejor_destino}")
    
    print(Fore.CYAN + "└" + Fore.GREEN + "─" * 78 + Style.RESET_ALL)
    
    # Mostrar interpretación
    print("\n" + Fore.YELLOW + "💡 Cómo interpretar estas recomendaciones:" + Style.RESET_ALL)
    print(Fore.WHITE + "Estas recomendaciones muestran hacia dónde es más probable que migren los usuarios ")
    print("de cada plataforma. Las integraciones con las plataformas destino podrían mejorar ")
    print("la retención de usuarios.")
    
    mostrar_pie()

def mostrar_vector_inicial():
    limpiar_consola()
    mostrar_encabezado("📊 VECTOR DE PROBABILIDAD INICIAL")
    print(Fore.CYAN + "Distribución de probabilidad inicial de las plataformas:\n")
    
    # Calcular el ancho máximo para la alineación
    max_len = max(len(plataforma) for plataforma in VECTOR_INICIAL.keys())
    
    # Mostrar cada plataforma con su probabilidad
    for plataforma, prob in VECTOR_INICIAL.items():
        barra = "█" * int(prob * 50)  # Barra de progreso
        print(f"{Fore.YELLOW}{plataforma.ljust(max_len)} {Fore.WHITE}[{prob:.2%}] {Fore.GREEN}{barra}")
    
    print(f"\n{Fore.CYAN}Total:{Fore.WHITE} {sum(VECTOR_INICIAL.values()):.0%}")
    mostrar_pie()
    

def mostrar_menu():
    limpiar_consola()
    print(Fore.CYAN + "╔" + "═" * 58 + "╗")
    print(Fore.CYAN + "║" + Fore.YELLOW + "                 MENÚ PRINCIPAL                  ".center(58) + Fore.CYAN + "║")
    print(Fore.CYAN + "╠" + "═" * 58 + "╣")
    
    opciones = [
        "1. Ver datos de navegación entre plataformas",
        "2. Mostrar matriz de transición",
        "3. Mostrar vector de probabilidad inicial",
        "4. Realizar pronóstico a n pasos",
        "5. Ver estado estacionario",
        "6. Obtener recomendaciones",
        "7. Salir"
    ]
    
    for opcion in opciones:
        print(Fore.CYAN + "║" + Fore.WHITE + f" {opcion.ljust(56)} " + Fore.CYAN + "║")
    
    print(Fore.CYAN + "╚" + "═" * 58 + "╝" + Style.RESET_ALL)

def mostrar_encabezado(titulo):
    limpiar_consola()
    print(Fore.GREEN + "╔" + "═" * 78 + "╗")
    print(Fore.GREEN + "║" + Fore.YELLOW + titulo.center(77) + Fore.GREEN + "║")
    print(Fore.GREEN + "╚" + "═" * 78 + "╝" + Style.RESET_ALL + "\n")

def mostrar_pie():
    print("\n" + Fore.CYAN + "═" * 60)
    input(Fore.WHITE + "\nPresiona " + Fore.GREEN + "Enter" + Fore.WHITE + " para continuar..." + Style.RESET_ALL)

def limpiar_consola():
    print("\n" * 50)

def main():
    while True:
        mostrar_menu()
        try:
            opcion = input("\n" + Fore.YELLOW + "➤ " + Fore.WHITE + "Selecciona una opción (1-7): " + Style.RESET_ALL)
            
            if opcion == "1":
                mostrar_datos_navegacion()
            elif opcion == "2":
                construir_matriz_transicion()
            elif opcion == "3":
                mostrar_vector_inicial()
            elif opcion == "4":
                pronosticar_probabilidad()
            elif opcion == "5":
                pronosticar_largo_plazo()
            elif opcion == "6":
                generar_recomendaciones()
            elif opcion == "7":
                print("\n" + Fore.GREEN + "╔" + "═" * 60 + "╗")
                print(Fore.GREEN + "║" + Fore.YELLOW + "¡Gracias por usar el sistema de análisis de IA!" + " " * 13 + Fore.GREEN + "║")
                print(Fore.GREEN + "║" + Fore.WHITE + "Saliendo del programa..." + " " * 36 + Fore.GREEN + "║")
                print(Fore.GREEN + "╚" + "═" * 60 + "╝" + Style.RESET_ALL)
                time.sleep(1)
                break
            else:
                print("\n" + Fore.RED + "❌ Opción no válida. Por favor, ingresa un número del 1 al 7." + Style.RESET_ALL)
                time.sleep(1)
        except Exception as e:
            print(f"\n{Fore.RED}❌ Ocurrió un error: {str(e)}{Style.RESET_ALL}")
            time.sleep(2)

if __name__ == "__main__":
    main()
