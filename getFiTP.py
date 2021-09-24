from paramiko import transport
import paramiko

fileName = "fc1d5596-ce14-4553-bd45-06da14438e9f.mp3"
t = paramiko.Transport("103.92.29.98")
t.connect(username="root",password="bbb1999@@")
sftp = paramiko.SFTPClient.from_transport(t)

sftp.chdir("/var/www/BBB_Backend/wwwroot/Temporary")
sftp.get(fileName,fileName)
sftp.close()