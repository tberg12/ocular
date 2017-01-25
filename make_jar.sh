cp lib/JCuda-All-0.6.0-bin-linux-x86_64/* lib/
cp lib/JCuda-All-0.6.0-bin-apple-x86_64/* lib/


java -Dfile.encoding=UTF8 -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256m -jar sbt-launch-*.jar "one-jar"
JARPATH=`expr target/scala-*/ocular_*-*-one-jar.jar`
FILENAME=$(basename $JARPATH)
VERSION=${FILENAME:12}
VERSION=${VERSION::${#VERSION}-12}
JARNAME="ocular-${VERSION}-with_dependencies.jar"
TEMPDIR=${FILENAME::${#FILENAME}-4}
mkdir $TEMPDIR
mv $JARPATH $TEMPDIR
cd $TEMPDIR
jar -xf $FILENAME
rm $FILENAME
cp ../lib/*.jar lib/
cp ../lib/JCuda-*/* lib/
jar cmf META-INF/MANIFEST.MF ../$JARNAME *
cd ..
rm -r $TEMPDIR

