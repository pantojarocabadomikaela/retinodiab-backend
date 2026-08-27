from django.contrib import admin
from .models import User
from .models import Diagnostico
from .models import Manual

# Register your models here.
admin.site.register(User)
admin.site.register(Diagnostico)
admin.site.register(Manual)