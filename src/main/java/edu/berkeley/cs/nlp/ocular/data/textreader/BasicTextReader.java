package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
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

		// Split characters and convert to diacritic-normalized forms.
		List<String> normalizedChars = new ArrayList<String>();
		for (String c : Charset.readNormalizeCharacters(line)) {
			normalizedChars.add(c);
		}
		return normalizedChars;
	}

	public String toString() {
		return "BasicTextReader()";
	}

}
