package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicTextReader implements TextReader {

	private Set<String> bannedChars;
	
	@SuppressWarnings("unchecked")
	public BasicTextReader() {
		this.bannedChars = Collections.EMPTY_SET;
	}
	
	public BasicTextReader(Set<String> bannedChars) {
		this.bannedChars = bannedChars;
	}
	
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
		for (String c : Charset.readCharacters(line)) {
			if (!bannedChars.contains(c)) {
				escapedChars.add(c);
			}
		}
		return escapedChars;
	}

	public String toString() {
		return "BasicTextReader()";
	}

}
