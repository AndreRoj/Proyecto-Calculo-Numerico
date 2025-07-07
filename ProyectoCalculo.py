from PIL import Image
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Fórmula para convertir un color RGB a su escala de grises
def rgb_to_grayscale(r, g, b):
    return int(0.299 * r + 0.587 * g + 0.114 * b)

# Función para cargar la imagen y convertirla a escala de grises
def convert_to_grayscale(image_path, grayscale_output_path):
    try:
        print(f"Abriendo imagen: {image_path}")
        image = Image.open(image_path)
        width, height = image.size
        print(f"Tamaño de la imagen: {width}x{height}")

        grayscale_image = Image.new('L', (width, height))

        print("Convirtiendo a escala de grises...")
        # Recorrido de cada pixel para convertirlo
        for x in range(width): # Eje X: izquierda → derecha
            for y in range(height): # Eje Y: arriba → abajo
                r, g, b = image.getpixel((x, y))
                gray_value = rgb_to_grayscale(r, g, b)
                grayscale_image.putpixel((x, y), gray_value)

        grayscale_image.save(grayscale_output_path)
        print(f"✅ Imagen en escala de grises guardada en: {grayscale_output_path}")

        # Llamar a la función para detección de bordes
        return apply_canny_edge_detector(grayscale_output_path, "bordes_detectados.jpg")

    except Exception as e:
        print(f"❌ Error: {e}")

def apply_canny_edge_detector(image_path, output_path):
    try:
        print(f"Aplicando detector de bordes Canny a: {image_path}")
        # Leer la imagen en escala de grises con OpenCV
        img = cv2.imread(image_path, 0)

        # Filtro gaussiano para reducir el ruido de la imagen
        blur = cv2.GaussianBlur(img, (5,5), 0)
        # Aplicar filtro Canny
        edges = cv2.Canny(blur, threshold1=150, threshold2=500)

        # Guardar la imagen resultante
        cv2.imwrite(output_path, edges)
        print(f"✅ Bordes detectados y guardados en: {output_path}")

        return edges

    except Exception as e:
        print(f"❌ Error al aplicar Canny: {e}")


def find_coordinates(edges, image_path):
        
        #se asegura de que sea binario
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        
        #Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("✅ Contornos detectados")

        #Filtrar y extraer el contorno superior (suponiendo que es el más alto en la imagen)
        if contours:
            # Ordenar contornos por posición vertical (y) mínima
            contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
            top_contour = contours_sorted[0]  # Contorno más arriba en la imagen

            # Extraer coordenadas (x, y) del contorno superior
            top_points = top_contour.squeeze()  # Elimina dimensiones innecesarias

            # Dibujar el contorno superior en verde para visualización
            imag = cv2.drawContours(image_path, [top_contour], -1, (0, 0, 255), 2)
            cv2.imwrite('contornos.png', imag)

            # Mostrar resultados
            cv2.imshow("Bordes (Canny)", edges)
            cv2.imshow("Contorno superior", imag)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            # Guardar coordenadas en un archivo o imprimirlas
            # print("Coordenadas del contorno superior:")
            # print(top_points)
            return top_contour.squeeze()
        
        else:
            print("No se encontraron contornos.")

def filter_coordinates(coordinates, image_path):
    coordenadas_np = np.array(coordinates)

    # Encontrar filas únicas para evitar redundancia
    coordenadas_f = np.unique(coordenadas_np, axis=0)
    # Encontrar columnas únicas ya que no es posible tener dos coordenadas con el mismo valor de x
    coordenadas_ordenadas = coordenadas_f[np.lexsort((coordenadas_f[:, 1], coordenadas_f[:, 0]))]
    _, indices_unicos = np.unique(coordenadas_ordenadas[:, 0], return_index=True)
    coordenadas_filtradas = coordenadas_ordenadas[indices_unicos]

    # print("Coordenadas filtradas (x único, y más pequeño):")
    # print(coordenadas_filtradas)

    # Mostrar la imagen con los puntos
    for (x, y) in coordenadas_filtradas:
        imag = cv2.circle(image_path, (x, y), radius=1, color=(0, 0, 255), thickness=-1)  # Puntos rojos
    
    cv2.imshow("Puntos filtrados (y más pequeño)", imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return coordenadas_filtradas

def interpolacion_splines_cubicos(coordinates):
    # Crear matriz de 3 filas × N columnas para los puntos(k, xk, yk)
    matrizpoint = np.zeros((3, len(coordinates)))  # Inicializar con ceros

    # Invertir los valores de Y para que coincidan con el sistema de coordenadas de Matplotlib
    coordinates[:, 1] = np.max(coordinates[:, 1]) - coordinates[:, 1]  # Inversión vertical

    # Llenar las filas para la matriz de puntos:
    matrizpoint[0, :] = np.arange(len(coordinates))  # valor de k de 0 a n numero de puntos
    matrizpoint[1, :] = coordinates[:,0] #valor de xk
    matrizpoint[2, :] = coordinates[:,1] #valor de yk

    # Crear matriz de 4 filas × n-1 columnas para los puntos(k, hk, λk, μk)
    matriz_elementos = np.zeros((4, len(coordinates)-1))  # Inicializar con ceros        

    #Calculos para hk, λk
    for i in range(0,len(coordinates)-1):
        #Calculos para hk, intervalo, hk = xk+1 − xk
        xkmas = matrizpoint[1, i+1]
        xk = matrizpoint[1, i]
        hk = xkmas - xk
        matriz_elementos[0, i] = hk
        # print(f'valor xk+1:{xkmas} y valor de xk:{xk}, total: {hk}')
        
        # Calculos para λk, λk = (yk+1 − yk)/hk
        ykmas = matrizpoint[2, i+1]
        yk = matrizpoint[2, i]
        landak = (ykmas - yk)/hk
        matriz_elementos[1, i] = landak

    #Calculos para μk
    for i in range(0,len(coordinates)-1):
        # Calculos para μk, μk = (λk − λk-1)*3
        if(i-1 >= 0):
            landk1 = matriz_elementos[1, i] #λk
            landkmenos = matriz_elementos[1, i-1] #λk-1
            muk = (landk1 - landkmenos)*3 #μk
            matriz_elementos[2, i] = muk

    #se pasa la matrizpoint con los datos de los puntos(k,xk,yk)
    #y la matriz_elementos con los datos valores calculadors de (hk,λk,μk)
    #nos retorna la matriz de borde de spline natural
    matriz_bordeSplineN = borde_spline_natural(matriz_elementos)
    matriz_resultante = resultados_sistema(matrizpoint,matriz_elementos)

    #X*C = R
    #con X (matriz_bordeSplineN) y R (matriz_resultante), se buscan los valores de C del Sistema
    C = np.linalg.solve(matriz_bordeSplineN, matriz_resultante)
    #Redondeado a 4 cifras decimales
    C_redondeado = np.round(C, decimals=4)

    spline = CubicSpline(matrizpoint[1,:], matrizpoint[2,:], bc_type='natural')

    # print("Coeficientes (a, b, c, d) por intervalo:")
    # print(spline.c)

    x_fine = np.linspace(min(matrizpoint[1, :]), max(matrizpoint[1, :]), 1000)
    y_fine = spline(x_fine)

    plt.ylim(0, 150)  # Límite inferior (y_min) y superior (y_max)
    # Graficar
    plt.plot(matrizpoint[1,:], matrizpoint[2,:], 'ro', label='Puntos originales')
    plt.plot(x_fine, y_fine, 'b-', label='Spline cúbico')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación con CubicSpline')
    plt.grid(True)
    plt.show()

    df_coord_element = pd.DataFrame(
        [
        ['--- COORDENADAS ---'],
        ["k"] + matrizpoint[0, :].tolist(),
        ["xk"] + matrizpoint[1, :].tolist(),
        ["yk"] + matrizpoint[2, :].tolist(),
        [''],
        ['--- Matriz Elementos ---'],
        ["hk"] + matriz_elementos[0, :].tolist(),
        ["λk"] + matriz_elementos[1, :].tolist(),
        ["μk"] + matriz_elementos[2, :].tolist(),
        ]
    )

    df_borde_splineNatu = pd.DataFrame(
        matriz_bordeSplineN
    )

    df_resultante = pd.DataFrame(
        matriz_resultante
    )

    df_C = pd.DataFrame(
        {
            'Variable': [f'c{i}' for i in range(len(C_redondeado))],
            'Valor': C_redondeado.flatten()
        }
    )

    # Exportar a Excel
    nombre_archivo = "matriz_puntos.xlsx"
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_coord_element.to_excel(writer, sheet_name='Coordenadas y Elementos', header=False, index=False)  # index=False evita una columna extra de índices
        df_borde_splineNatu.to_excel(writer, sheet_name='Borde Spline Natural', header=False, index=False)  # index=False evita una columna extra de índices
        df_resultante.to_excel(writer, sheet_name='Resultante Borde Spline Natural', header=False, index=False)  
        df_C.to_excel(writer, sheet_name='Valores C - ResulSist ', header=False, index=False)  

    print(f"✅ Datos exportados a '{nombre_archivo}'")

def borde_spline_natural(matriz_elementos): #S''(x0) = 0, S'' (xn) = 0,
    # Crear matriz de n filas × n columnas
    n = matriz_elementos.shape[1]
    matrizspline_natural = np.zeros((n,n))
    #print(n)
    for fila in range(0,n):
        for colum in range(0,n):
            if (fila==0 and colum==0) or (fila==n-1 and colum==n-1):
                matrizspline_natural[fila,colum] = 1
                #print('entro')
            elif (fila==colum):
                # print(f'f {fila} y c {colum}')
                # hk
                hk = matriz_elementos[0,colum-1]
                matrizspline_natural[fila,colum-1] = hk
                #print(f'h{fila-1} {hk}')
                # hk+1
                hkmas= matriz_elementos[0,colum]
                #print(f'h{fila} {hkmas}')
                matrizspline_natural[fila,colum+1] = hkmas
                #2(hk + hk+1)
                matrizspline_natural[fila,colum] = 2*(hk+hkmas)
                if(fila+1 < n-1):
                    matrizspline_natural[fila+1,colum] = hkmas
                if(fila-1 > 0):
                    matrizspline_natural[fila-1,colum] = hk

    #print(matrizspline_natural)
    return matrizspline_natural
                
def resultados_sistema(matriz_points,matriz_elementos):
    # Crear matriz de n filas × 1 columnas
    n = matriz_elementos.shape[1]
    matrizspline_natural = np.zeros((n,1))   

    for fila in range(0,n):
        if fila == 0 or fila == n-1:
            matrizspline_natural[fila]=0
        else:
            #(3(yk+2 - yk+1)/hk+1) - (3(yk+1 - yk)/hk)
            yk = matriz_points[2,fila-1] #yk
            ykmas = matriz_points[2,fila] #yk+1
            ykmas2 = matriz_points[2,fila+1] #yk+2
            hk = matriz_elementos[0,fila-1] #hk
            hkmas = matriz_elementos[0,fila] #hk+1
            r = (3*(ykmas2 - ykmas)/hkmas) - (3*(ykmas - yk)/hk)
            matrizspline_natural[fila] = r

    #print(matrizspline_natural)
    return matrizspline_natural


if __name__ == "__main__":
    input_image = "HulkProyecto.jpg"
    grayscale_image = "imagen_gris.jpg"
    print("=== Iniciando programa ===")
    edges_image = convert_to_grayscale(input_image, grayscale_image)
    coordinates = find_coordinates(edges_image, cv2.imread(input_image))
    coordinates_filter = filter_coordinates(coordinates, cv2.imread(input_image))
    interpolacion_splines_cubicos(coordinates_filter)
    print("=== Finalizado ===")


