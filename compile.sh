java -Dfile.encoding=UTF8 -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256m -jar sbt-launch-*.jar "one-jar"

FILENAME=`expr target/scala-*/ocular_*-*-one-jar.jar`
# echo $FILENAME
FILENAME=$(basename $FILENAME)
# echo $FILENAME
VERSION=${FILENAME:12}
# echo $VERSION
VERSION=${VERSION::${#VERSION}-12}
echo $VERSION

mv target/scala-*/ocular_*-*-one-jar.jar ./ocular-${VERSION}_with-dependencies.jar
# mv target/scala-*/ocular_*-*.jar ./ocular-${VERSION}.jar
