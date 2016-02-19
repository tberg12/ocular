import com.typesafe.sbt.SbtStartScript

import com.github.retronym.SbtOneJar._

name := "ocular-extend"

organization := "edu.berkeley.cs.nlp"

version := "0.2-SNAPSHOT"

scalaVersion := "2.11.7"

resolvers ++= Seq(
  "dhg releases repo" at "http://www.cs.utexas.edu/~dhg/maven-repository/releases",
  "dhg snapshot repo" at "http://www.cs.utexas.edu/~dhg/maven-repository/snapshots",
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases",
  "OpenNLP repo" at "http://opennlp.sourceforge.net/maven2"
)

javacOptions ++= Seq("-deprecation")

scalacOptions ++= Seq("-unchecked", "-deprecation")

seq(SbtStartScript.startScriptForClassesSettings: _*)

SbtStartScript.stage in Compile := Unit

mainClass := None

oneJarSettings

mainClass in oneJar := None

libraryDependencies ++= Seq(
  "dhg" % "scala-util_2.11" % "0.0.2-SNAPSHOT",
  "org.swinglabs" % "pdf-renderer" % "1.0.5",
  "junit" % "junit" % "4.12" % "test",
  "com.novocode" % "junit-interface" % "0.10" % "test")
