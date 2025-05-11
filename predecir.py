import flet as ft
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def main(page: ft.Page):
    # Configuración de página compacta
    page.title = "EmoScan"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 500  # Reducido de 650
    page.window_height = 600  # Reducido de 700
    page.window_resizable = False
    page.bgcolor = "#f5f7fa"
    
    # Carga el modelo
    model = load_model(r'C:/Users/gabri/Downloads/modelo_nuevo_L.h5')
    labels = ['ENOJADO', 'MIEDO', 'FELIZ', 'NEUTRAL', 'TRISTE', 'SORPRENDIDO']
    label_colors = {
        'ENOJADO': "#ff4757",
        'MIEDO': "#9c88ff",
        'FELIZ': "#4cd137",
        'NEUTRAL': "#487eb0",
        'TRISTE': "#3498db",
        'SORPRENDIDO': "#fbc531"
    }
    
    # Elementos de interfaz más compactos
    title = ft.Text(
        "EmoScan",
        size=24,  # Reducido de 32
        weight=ft.FontWeight.BOLD,
        color="#2f3640"
    )
    
    subtitle = ft.Text(
        "Detector de Emociones",
        size=14,  # Reducido de 16
        color="#7f8fa6"
    )
    
    image_container = ft.Container(
        width=350,  # Reducido de 450
        height=350,  # Reducido de 450
        border_radius=10,
        bgcolor="white",
        border=ft.border.all(1, "#dcdde1"),
        alignment=ft.alignment.center,
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Icon(ft.icons.IMAGE_SEARCH, size=40, color="#dcdde1"),
                ft.Text("Selecciona una imagen", size=14, color="#7f8fa6")
            ]
        )
    )
    
    result_container = ft.Container(
        width=350,  # Reducido de 450
        padding=15,
        border_radius=10,
        bgcolor="white",
        border=ft.border.all(1, "#dcdde1"),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("", size=18, weight=ft.FontWeight.BOLD),
                ft.Text("", size=14)
            ]
        )
    )
    
    def pick_files_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
            
        file_path = e.files[0].path
        
        # Mostrar imagen con tamaño ajustado
        image_container.content = ft.Image(
            src=file_path,
            width=330,  # Reducido de 430
            height=330,  # Reducido de 430
            fit=ft.ImageFit.CONTAIN,
            border_radius=8
        )
        page.update()
        
        # Preprocesamiento y predicción
        target_size = model.input_shape[1:3]
        img = image.load_img(file_path, target_size=target_size)
        x = image.img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        
        preds = model.predict(x)
        idx = np.argmax(preds[0])
        label = labels[idx]
        confidence = preds[0][idx] * 100
        
        # Mostrar resultados compactos
        result_container.content.controls[0] = ft.Text(
            f"Emoción: {label}",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=label_colors.get(label, "#487eb0")
        )
        result_container.content.controls[1] = ft.Text(
            f"Confianza: {confidence:.1f}%",
            size=14,
            color="#7f8fa6"
        )
        page.update()
    
    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)
    
    load_button = ft.ElevatedButton(
        "Cargar Imagen",
        icon=ft.icons.UPLOAD,
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["jpg", "jpeg", "png"]
        ),
        style=ft.ButtonStyle(
            padding=10,
            bgcolor="#487eb0",
            color="white"
        )
    )
    
    # Diseño compacto final
    page.add(
        ft.Column(
            spacing=15,  # Reducido de 30
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                title,
                subtitle,
                image_container,
                load_button,
                result_container,
                ft.Text("© EmoScan", size=10, color="#7f8fa6")
            ]
        )
    )

ft.app(target=main)