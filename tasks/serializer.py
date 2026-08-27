#para conectar con la tabla en la base de datos
from rest_framework import serializers
from .models import User
from .models import Diagnostico
from .models import Manual

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

class DiagnosticoSerializer(serializers.ModelSerializer):
    paciente_nombre = serializers.SerializerMethodField()

    class Meta:
        model = Diagnostico
        fields = '__all__'

    def get_paciente_nombre(self, obj):
        try:
            user = User.objects.get(email=obj.email)
            return user.nombre
        except User.DoesNotExist:
            return ''

class ManualSerializer(serializers.ModelSerializer):
    class Meta:
        model = Manual
        fields = '__all__'