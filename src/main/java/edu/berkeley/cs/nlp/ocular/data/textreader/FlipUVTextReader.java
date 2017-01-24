package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class FlipUVTextReader implements TextReader {

	private double flipRate;
	private TextReader delegate;
	
	private Random rand = new Random(0);

	public FlipUVTextReader(double flipRate, TextReader delegate) {
		this.flipRate = flipRate;
		this.delegate = delegate;
	}

	public List<String> readCharacters(String line) {
		List<String> chars = new ArrayList<String>();
		for (String c : delegate.readCharacters(line)) {
			if (c.equals("u")) {
				if (rand.nextDouble() < flipRate) {
					chars.add("u");
				} else {
					chars.add("v");
				}
			} else if (c.equals("U")) {
				if (rand.nextDouble() < flipRate) {
					chars.add("U");
				} else {
					chars.add("V");
				}
			} else if (c.equals("v")) {
				if (rand.nextDouble() < flipRate) {
					chars.add("v");
				} else {
					chars.add("u");
				}
			} else if (c.equals("V")) {
				if (rand.nextDouble() < flipRate) {
					chars.add("V");
				} else {
					chars.add("U");
				}
			} else {
				chars.add(c);
			}
		}
		return chars;
	}

	public String toString() {
		return "FlipUVTextReader(" + flipRate + ", " + delegate + ")";
	}

}
