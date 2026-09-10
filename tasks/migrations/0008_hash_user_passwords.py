from django.contrib.auth.hashers import identify_hasher, make_password
from django.db import migrations, models


def hash_existing_passwords(apps, schema_editor):
    User = apps.get_model('tasks', 'User')

    for user in User.objects.all().iterator():
        try:
            identify_hasher(user.password)
        except ValueError:
            user.password = make_password(user.password)
            user.save(update_fields=['password'])


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0007_alter_user_email'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='password',
            field=models.CharField(max_length=128),
        ),
        # El hash no es reversible: no es posible recuperar la contraseña original.
        migrations.RunPython(hash_existing_passwords),
    ]
