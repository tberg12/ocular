package edu.berkeley.cs.nlp.ocular.image;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

public class ImageUtils {
	
	// 255 so we don't get 256 when prob = 1.0
	public static final double MAX_LEVEL = 255;
	public static enum PixelType {BLACK, WHITE, OBSCURED};
	
	public static PixelType[][] getPixelTypes(BufferedImage image) {
		return getPixelTypes(getLevels(image));
	}
	
	public static PixelType[][] getPixelTypes(double[][] levels) {
		if (levels.length == 0) return new PixelType[0][0];
		PixelType[][] pixelTypes = new PixelType[levels.length][levels[0].length];
		for (int i=0; i<levels.length; ++i) {
			for (int j=0; j<levels[0].length; ++j) {
				double val = levels[i][j];
				if (val <= MAX_LEVEL / 2.0) {
					pixelTypes[i][j] = PixelType.BLACK;
				} else {
					pixelTypes[i][j] = PixelType.WHITE;
				}
			}
		}
		return pixelTypes;
	}
	
	public static PixelType getPixelType(double level) {
		if (level <= MAX_LEVEL / 2.0) {
			return PixelType.BLACK;
		} else {
			return PixelType.WHITE;
		}
	}
	
	public static double[][] getLevels(BufferedImage image) {
		BufferedImage grayImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);  
		Graphics g = grayImage.getGraphics();  
		g.drawImage(image, 0, 0, null);  
		g.dispose();
		Raster imageData = grayImage.getData();
    	double[][] levels = new double[imageData.getWidth()][imageData.getHeight()];
    	for (int i=0; i<imageData.getWidth(); ++i) {
    		for (int j=0; j<imageData.getHeight(); ++j) {
    			levels[i][j] = imageData.getPixel(i, j, (double[]) null)[0];
    		}
    	}
    	return levels;
	}
	
	public static BufferedImage makeImage(double[][] levels) {
		BufferedImage image = new BufferedImage(levels.length, levels[0].length, BufferedImage.TYPE_BYTE_GRAY);
		WritableRaster writeableRaster = image.getRaster();
		for (int i=0; i<writeableRaster.getWidth(); ++i) {
			for (int j=0; j<writeableRaster.getHeight(); ++j) {
				writeableRaster.setPixel(i, j, new double[] {levels[i][j]});
			}
		}
		return image;
	}
	
	public static BufferedImage makeRgbImage(int[][] rbgImage) {
		BufferedImage image = new BufferedImage(rbgImage.length, rbgImage[0].length, BufferedImage.TYPE_INT_RGB);
		for (int x=0; x<rbgImage.length; ++x) {
			for (int y=0; y<rbgImage[x].length; ++y) {
				image.setRGB(x, y, (int)rbgImage[x][y]);
			}
		}
		return image;
	}
	
	public static BufferedImage resampleImage(BufferedImage image, int height) {
		double mult = height / ((double) image.getHeight());
		Image unbufScaledImage = image.getScaledInstance((int)(mult * image.getWidth()), height, Image.SCALE_DEFAULT);
		BufferedImage scaledImage = new BufferedImage(unbufScaledImage.getWidth(null), unbufScaledImage.getHeight(null), BufferedImage.TYPE_BYTE_GRAY);
		Graphics g = scaledImage.createGraphics();
		g.drawImage(unbufScaledImage, 0, 0, null);
		g.dispose();
		return scaledImage;
	}

	public static BufferedImage rotateImage(BufferedImage image, double radians) {
		BufferedImage rotatedImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g2d = rotatedImage.createGraphics();
		g2d.rotate(radians);
		g2d.setBackground(Color.WHITE);
		int maxOfWidthHieght = Math.max(image.getWidth(null), image.getHeight(null));
		g2d.clearRect(-5*maxOfWidthHieght, -5*maxOfWidthHieght, 10*maxOfWidthHieght, 10*maxOfWidthHieght);
		g2d.drawImage(image, 0, 0, Color.WHITE, null);
		g2d.dispose();
		return rotatedImage;
	}
	
	public static BufferedImage cropImage(BufferedImage image, int lc, int rc, int tc, int bc) {
	    BufferedImage dest = new BufferedImage(image.getWidth()-(lc+rc), image.getHeight()-(tc+bc), BufferedImage.TYPE_BYTE_GRAY);
	    Graphics g = dest.getGraphics();
	    g.drawImage(image, 0, 0, image.getWidth()-(lc+rc), image.getHeight()-(tc+bc), lc, tc, image.getWidth()-rc, image.getHeight()-bc, null);
	    g.dispose();
	    return dest;
	}
	
	public static BufferedImage cropImageRelative(BufferedImage image, double leftCropFactor, double rightCropFactor, double topCropFactor, double bottomCropFactor) {
		int lc = (int) (leftCropFactor * image.getWidth());
		int rc = (int) (rightCropFactor * image.getWidth());
		int tc = (int) (topCropFactor * image.getHeight());
		int bc = (int) (bottomCropFactor * image.getHeight());
	    BufferedImage dest = new BufferedImage(image.getWidth()-(lc+rc), image.getHeight()-(tc+bc), BufferedImage.TYPE_BYTE_GRAY);
	    Graphics g = dest.getGraphics();
	    g.drawImage(image, 0, 0, image.getWidth()-(lc+rc), image.getHeight()-(tc+bc), lc, tc, image.getWidth()-rc, image.getHeight()-bc, null);
	    g.dispose();
	    return dest;
	}
	
	public static void display(final BufferedImage image) {
		final JFrame frame = new JFrame();
		frame.getContentPane().setLayout(new BorderLayout());

		final ImageIcon imageIcon = new ImageIcon(image);
		JLabel imageLabel = new JLabel(imageIcon);
		frame.getContentPane().add(new JScrollPane(imageLabel), BorderLayout.CENTER);
		//frame.getContentPane().add(new JLabel(new ImageIcon(img)));

		JPanel buttonPanel = new JPanel();
		buttonPanel.setLayout(new GridLayout(1,4));
		JButton zoomInHButton = new JButton("+H");
		JButton zoomOutHButton = new JButton("-H");
		JButton zoomInVButton = new JButton("+V");
		JButton zoomOutVButton = new JButton("-V");
		buttonPanel.add(zoomInHButton);
		buttonPanel.add(zoomOutHButton);
		buttonPanel.add(zoomInVButton);
		buttonPanel.add(zoomOutVButton);

		final AtomicInteger zoomX = new AtomicInteger(0);
		final AtomicInteger zoomY = new AtomicInteger(0);

		zoomInHButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				zoomX.set(zoomX.get() + 1);
				refreshViewer(image, zoomX, zoomY, imageIcon, frame);
			}
		});
		zoomOutHButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				zoomX.set(zoomX.get() - 1);
				refreshViewer(image, zoomX, zoomY, imageIcon, frame);
			}
		});
		zoomInVButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				zoomY.set(zoomY.get() + 1);
				refreshViewer(image, zoomX, zoomY, imageIcon, frame);
			}
		});
		zoomOutVButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				zoomY.set(zoomY.get() - 1);
				refreshViewer(image, zoomX, zoomY, imageIcon, frame);
			}
		});

		frame.getContentPane().add(buttonPanel, BorderLayout.SOUTH);

		frame.pack();
		frame.setVisible(true);
		frame.invalidate();
	}
	
	private static void refreshViewer(Image img, AtomicInteger zoomX, AtomicInteger zoomY, ImageIcon icon, Frame frame) {
		//System.err.println(zoomX);
		//System.err.println(zoomY);
		Image newImage;
		if (zoomX.get() == 1 && zoomY.get() == 1) {
			newImage = img;
		} else {
			newImage = img.getScaledInstance((int)(img.getWidth(frame) * Math.pow(2, zoomX.get())),
					(int)(img.getHeight(frame) * Math.pow(2, zoomY.get())),
					Image.SCALE_SMOOTH);
		}
		icon.setImage(newImage);
		frame.repaint();
	}
	
}
