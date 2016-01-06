package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class CharsetHelper {

	public static List<String> splitToEncodedWords(List<String> chars) {
		List<String> encodedWords = new ArrayList<String>();

		StringBuilder wordBuffer = new StringBuilder();
		StringBuilder puncBuffer = new StringBuilder();
		for (String c : chars) {
			if (Charset.SPACE.equals(c)) {
				if (wordBuffer.length() > 0) {
					encodedWords.add(wordBuffer.toString());
				}
				wordBuffer.setLength(0);
				puncBuffer.setLength(0);
			}
			else {
				String ec = Charset.escapeChar(c);
				if (Charset.isPunctuationChar(ec)) {
					if (wordBuffer.length() > 0) {
						puncBuffer.append(ec);
					}
				}
				else {
					wordBuffer.append(puncBuffer);
					wordBuffer.append(ec);
					puncBuffer.setLength(0);
				}
			}
		}
		if (wordBuffer.length() > 0) {
			encodedWords.add(wordBuffer.toString());
		}

		return encodedWords;
	}
}
