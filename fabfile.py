# fabfile.py
from fabric.contrib.files import append, exists, sed, put
from fabric.api import env, local, run, sudo
import os
import json

# 현재 fabfile.py가 있는 폴더의 경로
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# deploy.json이라는 파일을 열어 아래의 변수들에 담아줍니다.
envs = json.load(open(os.path.join(PROJECT_DIR, "deploy.json")))

REPO_URL = envs['REPO_URL']
PROJECT_NAME = envs['PROJECT_NAME']
REMOTE_HOST = envs['REMOTE_HOST']
REMOTE_HOST_SSH = envs['REMOTE_HOST_SSH']
REMOTE_USER = envs['REMOTE_USER']

# SSH에 접속할 유저를 지정하고,
env.user = REMOTE_USER
# SSH로 접속할 서버주소를 넣어주고,
env.hosts = [
    REMOTE_HOST_SSH,
]
# 원격 서버중 어디에 프로젝트를 저장할지 지정해준 뒤,
project_folder = '/home/{}/{}'.format(env.user, PROJECT_NAME)
# 우리 프로젝트에 필요한 apt 패키지들을 적어줍니다.
apt_requirements = [
    'curl',
    'git',
    'python3-dev',
    'python3-pip',
    'build-essential',
    'apache2',
    'libapache2-mod-wsgi-py3',
    'python3-setuptools',
    'libssl-dev',
    'libffi-dev',
]


# _로 시작하지 않는 함수들은 fab new_server 처럼 명령줄에서 바로 실행이 가능합니다.
def new_server():
    setup()
    deploy()


def setup():
    _get_latest_apt()
    _install_apt_requirements(apt_requirements)
    _make_virtualenv()


def deploy():
    _get_latest_source()
    _put_envs()
    _update_virtualenv()
    _make_virtualhost()
    _grant_apache2()
    _restart_apache2()


# put이라는 방식으로 로컬의 파일을 원격지로 업로드할 수 있습니다.
def _put_envs():
    pass  # activate for envs.json file
    # put('envs.json', '~/{}/envs.json'.format(PROJECT_NAME))


# apt 패키지를 업데이트 할 지 결정합니다.
def _get_latest_apt():
    update_or_not = input('would you update?: [y/n]')
    if update_or_not == 'y':
        sudo('apt-get update && apt-get -y upgrade')


# 필요한 apt 패키지를 설치합니다.
def _install_apt_requirements(apt_requirements):
    reqs = ''
    for req in apt_requirements:
        reqs += (' ' + req)
    sudo('apt-get -y install {}'.format(reqs))


# virtualenv와 virtualenvwrapper를 받아 설정합니다.
def _make_virtualenv():
    if not exists('~/.virtualenvs'):
        script = '''"# python virtualenv settings
                    export WORKON_HOME=~/.virtualenvs
                    export VIRTUALENVWRAPPER_PYTHON="$(command \which python3)"  # location of python3
                    source /usr/local/bin/virtualenvwrapper.sh"'''
        run('mkdir ~/.virtualenvs')
        sudo('pip3 install virtualenv virtualenvwrapper')
        run('echo {} >> ~/.bashrc'.format(script))


# Git Repo에서 최신 소스를 받아옵니다.
# 깃이 있다면 fetch를, 없다면 clone을 진행합니다.
def _get_latest_source():
    if exists(project_folder + '/.git'):
        run('cd %s && git fetch' % (project_folder,))
    else:
        run('git clone %s %s' % (REPO_URL, project_folder))
    current_commit = local("git log -n 1 --format=%H", capture=True)
    run('cd %s && git reset --hard %s' % (project_folder, current_commit))


# Repo에서 받아온 requirements.txt를 통해 pip 패키지를 virtualenv에 설치해줍니다.
def _update_virtualenv():
    virtualenv_folder = project_folder + '/../.virtualenvs/{}'.format(PROJECT_NAME)
    if not exists(virtualenv_folder + '/bin/pip'):
        run('cd /home/%s/.virtualenvs && virtualenv %s' % (env.user, PROJECT_NAME))
    run('%s/bin/pip install -r %s/requirements.txt' % (
        virtualenv_folder, project_folder
    ))


# (optional) UFW에서 80번/tcp포트를 열어줍니다.
def _ufw_allow():
    sudo("ufw allow 'Apache Full'")
    sudo("ufw reload")


# Apache2의 Virtualhost를 설정해 줍니다.
# 이 부분에서 wsgi.py와의 통신, 그리고 virtualenv 내의 파이썬 경로를 지정해 줍니다.
def _make_virtualhost():
    script = """'<VirtualHost *:80>
    ServerName {servername}
    <Directory /home/{username}/{project_name}>
        <Files wsgi.py>
            Require all granted
        </Files>
    </Directory>
    WSGIDaemonProcess {project_name} python-home=/home/{username}/.virtualenvs/{project_name} python-path=/home/{username}/{project_name}
    WSGIProcessGroup {project_name}
    WSGIScriptAlias / /home/{username}/{project_name}/wsgi.py

    ErrorLog ${{APACHE_LOG_DIR}}/error.log
    CustomLog ${{APACHE_LOG_DIR}}/access.log combined

    </VirtualHost>'""".format(
        username=REMOTE_USER,
        project_name=PROJECT_NAME,
        servername=REMOTE_HOST,
    )
    sudo('echo {} > /etc/apache2/sites-available/{}.conf'.format(script, PROJECT_NAME))
    sudo('a2ensite {}.conf'.format(PROJECT_NAME))


# Apache2가 프로젝트 파일을 읽을 수 있도록 권한을 부여합니다.
def _grant_apache2():
    sudo('chown -R :www-data ~/{}'.format(PROJECT_NAME))
    sudo('chmod -R 775 ~/{}'.format(PROJECT_NAME))


# 마지막으로 Apache2를 재시작합니다.
def _restart_apache2():
    sudo('sudo service apache2 restart')
