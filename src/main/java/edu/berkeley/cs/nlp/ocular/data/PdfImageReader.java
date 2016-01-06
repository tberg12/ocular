package edu.berkeley.cs.nlp.ocular.data;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import com.sun.pdfview.PDFFile;
import com.sun.pdfview.PDFPage;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class PdfImageReader {

	public static int numPagesInPdf(File pdfFile) {
		try {
			RandomAccessFile raf = new RandomAccessFile(pdfFile, "r");
			FileChannel channel = raf.getChannel();
			ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
			PDFFile pdf = new PDFFile(buf);
			int numPages = pdf.getNumPages();
			raf.close();
			return numPages;
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static List<BufferedImage> readPdfAsImages(File pdfFile) {
		try {
			RandomAccessFile raf = new RandomAccessFile(pdfFile, "r");
			FileChannel channel = raf.getChannel();
			ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
			PDFFile pdf = new PDFFile(buf);

			List<BufferedImage> images = new ArrayList<BufferedImage>();
			for (int pageNumber = 1; pageNumber <= pdf.getNumPages(); ++pageNumber) {
				images.add(readPage(pdf, pageNumber));
			}

			raf.close();
			return images;
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * 
	 * @param pdfFile
	 *          Path to the pdf file.
	 * @param pageNumber
	 *          One-based page number to read
	 * @return
	 */
	public static BufferedImage readPdfPageAsImage(File pdfFile, int pageNumber) {
		if (pageNumber < 1)
			throw new RuntimeException("page numbering starts with 1; '" + pageNumber + "' given");
		try {
			RandomAccessFile raf = new RandomAccessFile(pdfFile, "r");
			FileChannel channel = raf.getChannel();
			ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
			PDFFile pdf = new PDFFile(buf);
			BufferedImage image = readPage(pdf, pageNumber);
			raf.close();
			return image;
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static BufferedImage readPage(PDFFile pdf, int pageNumber) {
		double scale = 2.5; // because otherwise the image comes out really tiny
		PDFPage page = pdf.getPage(pageNumber);
		Rectangle rect = new Rectangle(0, 0, (int) page.getBBox().getWidth(), (int) page.getBBox().getHeight());
		BufferedImage bufferedImage = new BufferedImage((int)(rect.width * scale), (int)(rect.height * scale), BufferedImage.TYPE_INT_RGB);
		Image image = page.getImage((int)(rect.width * scale), (int)(rect.height * scale), rect, null, true, true);
		Graphics2D bufImageGraphics = bufferedImage.createGraphics();
		bufImageGraphics.drawImage(image, 0, 0, null);
		return bufferedImage;
	}
}

