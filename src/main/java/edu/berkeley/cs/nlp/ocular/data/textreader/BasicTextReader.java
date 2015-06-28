package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicTextReader implements TextReader {

	public List<List<String>> readCharacters(List<String> lines) {
		List<List<String>> characterLines = new ArrayList<List<String>>();
		for (String l : lines)
			characterLines.add(readCharacters(l));
		return characterLines;
	}

	public List<String> readCharacters(String line) {
		line = line.replaceAll("``", "\"");
		line = line.replaceAll("''", "\"");
		line = line.replaceAll("\t", "    ");

		/*
		 * Split characters and replace diacritics with either diacritic codes or
		 * diacritic-less letters.
		 */
		List<String> escapedChars = new ArrayList<String>();
		int i = 0;
		while (i < line.length()) {
			Tuple2<String, Integer> escapedCharAndLength = Charset.readCharAt(line, i);
			String c = escapedCharAndLength._1;
			int length = escapedCharAndLength._2;
			escapedChars.add(c);
			i += length; // advance to the next character
		}
		return escapedChars;
	}

	public String toString() {
		return "BasicTextReader()";
	}

}
