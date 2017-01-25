import com.typesafe.sbt.SbtStartScript

import com.github.retronym.SbtOneJar._

name := "ocular"

organization := "edu.berkeley.cs.nlp"

version := "0.3-SNAPSHOT"

scalaVersion := "2.12.1"

javacOptions ++= Seq("-source", "1.6", "-target", "1.6")

Seq(SbtStartScript.startScriptForClassesSettings: _*)

SbtStartScript.stage in Compile := Unit

oneJarSettings

mainClass in oneJar := None


libraryDependencies ++= Seq(
//  "org.apache.commons" % "commons-lang3" % "3.4", //to escape HTML special characters
  "org.swinglabs" % "pdf-renderer" % "1.0.5",
  "junit" % "junit" % "4.12" % "test",
  "com.novocode" % "junit-interface" % "0.10" % "test")
