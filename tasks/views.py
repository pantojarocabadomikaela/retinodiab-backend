from rest_framework import viewsets
from .serializer import UserSerializer
from .serializer import DiagnosticoSerializer
from .serializer import ManualSerializer
from .models import User
from .models import Diagnostico
from .models import Manual
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt

import io
import base64
import datetime

# AI imports and model loaded lazily (only when evaluateImage is called)
_modelo = None
_model_labels = ['No retinopatia', 'Retinopatia E1', 'Retinopatia E2', 'Retinopatia E3']

MODEL_PATH = r"tasks\iamodels\diabetes_4.h5"
HEIGHT = 320
WIDTH = 320


def _get_model():
    global _modelo
    if _modelo is None:
        import numpy as np
        import tensorflow as tf
        from keras.models import load_model
        custom_objects = {'Optimizer': tf.optimizers.Adam}
        _modelo = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        _modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return _modelo


# Create your views here.
class UserView(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()

    @csrf_exempt
    @staticmethod
    def validateCredentials(request):
        if request.method == 'POST':
            email = request.POST.get('email')
            password = request.POST.get('password')
            user = User.authenticate(email, password)
            if user is not None:
                return JsonResponse({
                    'id': user.id,
                    'email': user.email,
                    'nombre': user.nombre,
                    'rol': user.rol,
                    'diabetes': user.diabetes,
                    'fecha_nacimiento': user.fecha_nacimiento.isoformat() if user.fecha_nacimiento else None
                }, status=200)
            else:
                return JsonResponse({'mensaje': 'Credenciales inválidas'}, status=400)
        else:
            return JsonResponse({'error': 'Método no permitido'}, status=405)

    @csrf_exempt
    @staticmethod
    def evaluateImage(request):
        if request.method == 'POST':
            try:
                import numpy as np
                import cv2
                from keras.preprocessing.image import img_to_array
            except ImportError:
                return JsonResponse({'error': 'Dependencias de IA no instaladas (tensorflow, opencv)'}, status=503)

            modelo = _get_model()

            imageFile = request.FILES.get('imageFile')
            email = request.POST.get('email')
            observaciones = request.POST.get('observaciones')

            if imageFile is None:
                return JsonResponse({'error': 'No se recibió ninguna imagen'}, status=400)

            image_bytes = imageFile.read()
            image_in_memory = io.BytesIO(image_bytes)
            image_in_memory.name = imageFile.name

            predicciones = UserView.model_prediction(image_in_memory, modelo)
            resultado = _model_labels[np.argmax(predicciones)]

            fecha_actual = datetime.datetime.now()
            cadena_fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
            nombre_imagen = "imagen_" + cadena_fecha + "_resultado_" + resultado

            image_in_memory.seek(0)
            image_base64 = base64.b64encode(image_in_memory.read()).decode('utf-8')

            diagnostico = Diagnostico(
                email=email,
                nombre=nombre_imagen,
                resultado=resultado,
                observaciones=observaciones,
                imagen=image_base64
            )
            diagnostico.save()

            return JsonResponse({'mensaje': resultado, 'nombre': nombre_imagen}, status=200)
        else:
            return JsonResponse({'error': 'Método no permitido'}, status=404)

    @staticmethod
    def crop_image_from_gray(img, tol=7):
        import numpy as np
        import cv2
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    @staticmethod
    def circle_crop(img, sigmaX=30):
        import cv2
        import numpy as np
        img = UserView.crop_image_from_gray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, depth = img.shape
        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = UserView.crop_image_from_gray(img)
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
        return img

    @staticmethod
    def preprocess_image(image_file):
        import numpy as np
        import cv2
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = UserView.circle_crop(img)
        return cv2.resize(img, (WIDTH, HEIGHT))

    @staticmethod
    def model_prediction(img_path, model):
        from keras.preprocessing.image import img_to_array
        img_preprocess = UserView.preprocess_image(img_path)
        image_array = img_to_array(img_preprocess)
        x = image_array.reshape((1,) + image_array.shape)
        x = x / 255.0
        pred = model.predict(x)
        return pred


class DiagnosticoView(viewsets.ModelViewSet):
    serializer_class = DiagnosticoSerializer
    queryset = Diagnostico.objects.all()


class ManualView(viewsets.ModelViewSet):
    serializer_class = ManualSerializer
    queryset = Manual.objects.all()
