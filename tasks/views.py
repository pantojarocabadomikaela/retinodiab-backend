from rest_framework import viewsets
from .serializer import UserSerializer
from .serializer import DiagnosticoSerializer
from .serializer import ManualSerializer
from .models import User
from .models import Diagnostico
from .models import Manual
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import base64
import datetime

# P0 RGB EfficientNetB0 Fold 4 — PyTorch inference
from .ai.inference import predict_image, InferenceError
from .ai.preprocessing import ImagePreprocessingError
from .ai.model_loader import ModelLoadingError


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
        """
        Recibe una retinografía, ejecuta inferencia con el modelo
        P0 RGB EfficientNetB0 Fold 4 y guarda el diagnóstico.

        El frontend existente mantiene compatibilidad con:
            - mensaje
            - nombre

        Además se devuelven:
            - prediction.class_id
            - prediction.label
            - probabilities
            - model_id
            - device
        """
        if request.method != 'POST':
            return JsonResponse({'error': 'Método no permitido'}, status=405)

        imageFile = request.FILES.get('imageFile')
        email = request.POST.get('email')
        observaciones = request.POST.get('observaciones')

        if imageFile is None:
            return JsonResponse(
                {'error': 'No se recibió ninguna imagen'},
                status=400
            )

        try:
            # Leer una sola vez: los mismos bytes se usan para IA y almacenamiento.
            image_bytes = imageFile.read()

            # Inferencia PyTorch:
            # image bytes
            #   -> preprocessing P0
            #   -> EfficientNetB0 Fold 4
            #   -> logits
            #   -> softmax
            #   -> clase + probabilidades
            result = predict_image(image_bytes)

        except ImagePreprocessingError as exc:
            return JsonResponse(
                {'error': str(exc)},
                status=400
            )

        except FileNotFoundError as exc:
            # Checkpoint no disponible en la ruta esperada.
            return JsonResponse(
                {
                    'error': 'No se encontró el modelo de IA configurado.',
                    'detalle': str(exc)
                },
                status=503
            )

        except ModelLoadingError as exc:
            return JsonResponse(
                {
                    'error': 'No se pudo cargar el modelo de IA.',
                    'detalle': str(exc)
                },
                status=503
            )

        except InferenceError as exc:
            return JsonResponse(
                {
                    'error': 'Falló la inferencia del modelo.',
                    'detalle': str(exc)
                },
                status=500
            )

        except Exception as exc:
            # Evita que un fallo inesperado termine en una respuesta HTML de Django.
            # Durante desarrollo incluimos el detalle para facilitar depuración.
            return JsonResponse(
                {
                    'error': 'Error inesperado al procesar la imagen.',
                    'detalle': str(exc)
                },
                status=500
            )

        resultado = result.label

        fecha_actual = datetime.datetime.now()
        cadena_fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
        nombre_imagen = (
            "imagen_"
            + cadena_fecha
            + "_resultado_"
            + resultado
        )

        # Conserva el comportamiento del sistema existente:
        # la imagen original se almacena codificada en base64.
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        diagnostico = Diagnostico(
            email=email,
            nombre=nombre_imagen,
            resultado=resultado,
            observaciones=observaciones,
            imagen=image_base64
        )
        diagnostico.save()

        # Conservamos 'mensaje' y 'nombre' para no romper el frontend anterior.
        # Añadimos información más rica de la inferencia.
        response_data = {
            'mensaje': resultado,
            'nombre': nombre_imagen,
            'prediction': {
                'class_id': result.class_id,
                'label': result.label
            },
            'probabilities': result.probabilities_by_label(),
            'model_id': result.model_id,
            'device': result.device
        }

        return JsonResponse(response_data, status=200)


class DiagnosticoView(viewsets.ModelViewSet):
    serializer_class = DiagnosticoSerializer
    queryset = Diagnostico.objects.all()


class ManualView(viewsets.ModelViewSet):
    serializer_class = ManualSerializer
    queryset = Manual.objects.all()
