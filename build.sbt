import com.typesafe.sbt.SbtStartScript

import com.github.retronym.SbtOneJar._

name := "ocular"

organization := "edu.berkeley.cs.nlp"

version := "0.3-SNAPSHOT"

scalaVersion := "2.11.7"

seq(SbtStartScript.startScriptForClassesSettings: _*)

SbtStartScript.stage in Compile := Unit

oneJarSettings

mainClass in oneJar := None


libraryDependencies ++= Seq(
  "org.swinglabs" % "pdf-renderer" % "1.0.5",
  "junit" % "junit" % "4.12" % "test",
  "com.novocode" % "junit-interface" % "0.10" % "test")
