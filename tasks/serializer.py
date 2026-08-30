#para conectar con la tabla en la base de datos
from rest_framework import serializers
from .models import User
from .models import Diagnostico
from .models import Manual

class UserSerializer(serializers.ModelSerializer):
    email = serializers.CharField(validators=[])

    def validate_email(self, value):
        queryset = User.objects.filter(email=value)
        if self.instance is not None:
            queryset = queryset.exclude(pk=self.instance.pk)

        if queryset.exists():
            raise serializers.ValidationError('email duplicado')
        return value

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