from PIL import Image
import cv2
import numpy as np
import pandas as pd

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

            # Opcional: Dibujar el contorno superior en verde para visualización
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

    print("Coordenadas filtradas (x único, y más pequeño):")
    print(coordenadas_filtradas)

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

    # Llenar las filas para la matriz de puntos:
    matrizpoint[0, :] = np.arange(len(coordinates))  # valor de k de 0 a n numero de puntos
    matrizpoint[1, :] = coordinates[:,0] #valor de xk
    matrizpoint[2, :] = coordinates[:,1] #valor de yk

    # Crear matriz de 4 filas × N columnas para los puntos(k, hk, λk, μk)
    matriz = np.zeros((4, len(coordinates)))  # Inicializar con ceros        

    #Calculos para hk, λk, μk
    for i in range(0,len(coordinates)-1):
        #Calculos para hk, intervalo, hk = xk+1 − xk
        xkmas = matrizpoint[1, i+1]
        xk = matrizpoint[1, i]
        hk = xkmas - xk
        matriz[0, i] = hk
        # print(f'valor xk+1:{xkmas} y valor de xk:{xk}, total: {hk}')
        
        # Calculos para λk, λk = (yk+1 − yk)/hk
        ykmas = matrizpoint[2, i+1]
        yk = matrizpoint[2, i]
        landak = (ykmas - yk)/hk
        matriz[1, i] = landak

        # Calculos para μk


    df = pd.DataFrame(
        [["k"] + matrizpoint[0, :].tolist(),
        ["xk"] + matrizpoint[1, :].tolist(),
        ["yk"] + matrizpoint[2, :].tolist(),
        ["hk"] + matriz[0, :].tolist(),
        ["λk"] + matriz[1, :].tolist(),
        ]
    )

    # Exportar a Excel
    nombre_archivo = "matriz_puntos.xlsx"
    df.to_excel(nombre_archivo, header=False, index=False)  # index=False evita una columna extra de índices

    print(f"✅ Datos exportados a '{nombre_archivo}'")

             


if __name__ == "__main__":
    input_image = "HulkProyecto.jpg"
    grayscale_image = "imagen_gris.jpg"
    print("=== Iniciando programa ===")
    edges_image = convert_to_grayscale(input_image, grayscale_image)
    coordinates = find_coordinates(edges_image, cv2.imread(input_image))
    coordinates_filter = filter_coordinates(coordinates, cv2.imread(input_image))
    interpolacion_splines_cubicos(coordinates_filter)
    print("=== Finalizado ===")


