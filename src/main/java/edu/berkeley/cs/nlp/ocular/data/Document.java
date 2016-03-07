package edu.berkeley.cs.nlp.ocular.data;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

public interface Document {
	public String baseName();
	public PixelType[][][] loadLineImages();
	public String[][] loadLineText();
	public List<String> loadLmText();
}
