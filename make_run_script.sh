cp lib/JCuda-All-0.6.0-bin-linux-x86_64/* lib/
cp lib/JCuda-All-0.6.0-bin-apple-x86_64/* lib/


java -Dfile.encoding=UTF8 -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256m -jar sbt-launch-*.jar "start-script"
