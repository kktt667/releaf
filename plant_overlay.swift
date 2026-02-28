import Cocoa

enum Stage: String {
    case normal
    case warning
    case decay
    case recovery
}

class OverlayWindow: NSWindow {
    init() {
        super.init(
            contentRect: NSScreen.main?.frame ?? .zero,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )
        backgroundColor = .clear
        isOpaque = false
        level = .screenSaver
        ignoresMouseEvents = true
        collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        contentView = OverlayView(frame: NSScreen.main?.frame ?? .zero)
        orderFrontRegardless()
    }
}

class OverlayView: NSView {
    var stage: Stage = .normal
    var lastStageValue = ""
    var pulse: CGFloat = 0.0

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            self?.checkStageFile()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func tick() {
        pulse += 0.06
        if pulse > 1000 { pulse = 0 }
        needsDisplay = true
    }

    func checkStageFile() {
        let path = FileManager.default.currentDirectoryPath + "/stage.txt"
        guard
            let value = try? String(contentsOfFile: path, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
            value != lastStageValue,
            let parsed = Stage(rawValue: value)
        else { return }

        lastStageValue = value
        stage = parsed
        print("[overlay] stage -> \(value)")
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width
        let h = bounds.height

        // Soft stage effects first, then widget cards.
        drawStageEffects(ctx: ctx, w: w, h: h)
        drawEyeCard(ctx: ctx, x: 18, y: h - 140, w: 210, h: 118)
        drawStateCard(ctx: ctx, x: 18, y: h - 278, w: 260, h: 120)
        drawRecoveryCard(ctx: ctx, x: w - 292, y: h - 188, w: 270, h: 166)
    }

    func stageColor() -> NSColor {
        switch stage {
        case .normal: return NSColor(calibratedRed: 0.38, green: 0.73, blue: 0.53, alpha: 0.95)
        case .warning: return NSColor(calibratedRed: 0.93, green: 0.70, blue: 0.37, alpha: 0.95)
        case .decay: return NSColor(calibratedRed: 0.84, green: 0.53, blue: 0.60, alpha: 0.95)
        case .recovery: return NSColor(calibratedRed: 0.46, green: 0.70, blue: 0.94, alpha: 0.95)
        }
    }

    func stageText() -> String {
        switch stage {
        case .normal: return "Calm mode"
        case .warning: return "Warning mode"
        case .decay: return "Soft decay mode"
        case .recovery: return "Healing mode"
        }
    }

    func drawStageEffects(ctx: CGContext, w: CGFloat, h: CGFloat) {
        let phase = (sin(pulse) + 1.0) * 0.5
        let center = CGPoint(x: w * 0.5, y: h * 0.5)
        let radius = max(w, h) * 0.7

        switch stage {
        case .normal:
            // Barely-there neutral tint
            ctx.setFillColor(NSColor(calibratedWhite: 0.0, alpha: 0.04).cgColor)
            ctx.fill(bounds)
        case .warning:
            let colors = [
                NSColor(calibratedRed: 1.0, green: 0.78, blue: 0.35, alpha: 0.03).cgColor,
                NSColor(calibratedRed: 1.0, green: 0.64, blue: 0.18, alpha: 0.09).cgColor,
            ] as CFArray
            let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0.0, 1.0])!
            ctx.drawRadialGradient(g, startCenter: center, startRadius: 0, endCenter: center, endRadius: radius, options: .drawsAfterEndLocation)
        case .decay:
            // Softer, theme-matching decay (not harsh red).
            let edgeAlpha = 0.10 + (0.06 * phase)
            let colors = [
                NSColor(calibratedRed: 0.80, green: 0.50, blue: 0.52, alpha: 0.03).cgColor,
                NSColor(calibratedRed: 0.55, green: 0.26, blue: 0.34, alpha: edgeAlpha).cgColor,
            ] as CFArray
            let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0.0, 1.0])!
            ctx.drawRadialGradient(g, startCenter: center, startRadius: 0, endCenter: center, endRadius: radius, options: .drawsAfterEndLocation)
            // Gentle vignette ring
            ctx.setStrokeColor(NSColor(calibratedRed: 0.67, green: 0.34, blue: 0.45, alpha: 0.24).cgColor)
            ctx.setLineWidth(10)
            ctx.stroke(bounds.insetBy(dx: 8, dy: 8))
        case .recovery:
            // Pleasant healing sweep + cool glow.
            let glow = 0.12 + (0.08 * phase)
            let colors = [
                NSColor(calibratedRed: 0.45, green: 0.90, blue: 0.88, alpha: 0.04).cgColor,
                NSColor(calibratedRed: 0.20, green: 0.66, blue: 0.95, alpha: glow).cgColor,
            ] as CFArray
            let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0.0, 1.0])!
            ctx.drawRadialGradient(g, startCenter: center, startRadius: 0, endCenter: center, endRadius: radius, options: .drawsAfterEndLocation)

            // Soft moving highlight stripe
            let x = (w + 220) * phase - 220
            let stripeRect = CGRect(x: x, y: 0, width: 180, height: h)
            ctx.saveGState()
            ctx.clip(to: bounds)
            ctx.setFillColor(NSColor(calibratedRed: 0.75, green: 1.0, blue: 0.95, alpha: 0.08).cgColor)
            ctx.fill(stripeRect)
            ctx.restoreGState()
        }
    }

    func drawCardBackground(ctx: CGContext, rect: CGRect) {
        let path = CGPath(roundedRect: rect, cornerWidth: 16, cornerHeight: 16, transform: nil)
        ctx.addPath(path)
        ctx.setFillColor(NSColor(calibratedRed: 0.12, green: 0.14, blue: 0.22, alpha: 0.72).cgColor)
        ctx.fillPath()
        ctx.addPath(path)
        ctx.setStrokeColor(NSColor(calibratedRed: 0.52, green: 0.56, blue: 0.74, alpha: 0.55).cgColor)
        ctx.setLineWidth(1.5)
        ctx.strokePath()
    }

    func drawEyeCard(ctx: CGContext, x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat) {
        let rect = CGRect(x: x, y: y, width: w, height: h)
        drawCardBackground(ctx: ctx, rect: rect)
        drawLabel("Focus Buddy", at: CGPoint(x: x + 14, y: y + h - 30), size: 15, color: .white, weight: .semibold)
        drawLabel("Tiny camera view shows focus status", at: CGPoint(x: x + 14, y: y + h - 52), size: 11, color: .lightGray, weight: .regular)
        let eyeColor = (stage == .decay) ? NSColor.systemRed : NSColor.systemGreen
        ctx.setFillColor(eyeColor.cgColor)
        ctx.fillEllipse(in: CGRect(x: x + 14, y: y + 20, width: 14, height: 14))
        drawLabel("Eyes visibility indicator", at: CGPoint(x: x + 36, y: y + 20), size: 11, color: .white, weight: .regular)
    }

    func drawStateCard(ctx: CGContext, x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat) {
        let rect = CGRect(x: x, y: y, width: w, height: h)
        drawCardBackground(ctx: ctx, rect: rect)
        drawLabel("Garden State", at: CGPoint(x: x + 14, y: y + h - 30), size: 15, color: .white, weight: .semibold)
        drawLabel(stage.rawValue.uppercased(), at: CGPoint(x: x + 14, y: y + h - 62), size: 22, color: stageColor(), weight: .bold)
        drawLabel(stageText(), at: CGPoint(x: x + 14, y: y + 26), size: 12, color: .lightGray, weight: .regular)
    }

    func drawRecoveryCard(ctx: CGContext, x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat) {
        let rect = CGRect(x: x, y: y, width: w, height: h)
        drawCardBackground(ctx: ctx, rect: rect)
        drawLabel("Recovery Guide", at: CGPoint(x: x + 14, y: y + h - 30), size: 15, color: .white, weight: .semibold)

        let phase = (sin(pulse) + 1.0) * 0.5
        let ringColor = stageColor().withAlphaComponent(0.35 + (0.35 * phase))
        ctx.setStrokeColor(ringColor.cgColor)
        ctx.setLineWidth(5)
        ctx.strokeEllipse(in: CGRect(x: x + w - 78, y: y + h - 84, width: 52, height: 52))

        drawLabel("1) Focus on screen for tracking", at: CGPoint(x: x + 14, y: y + h - 62), size: 11, color: .lightGray, weight: .regular)
        drawLabel("2) Press Y when prompted", at: CGPoint(x: x + 14, y: y + h - 82), size: 11, color: .lightGray, weight: .regular)
        drawLabel("3) Upload an outdoor photo", at: CGPoint(x: x + 14, y: y + h - 102), size: 11, color: .lightGray, weight: .regular)
        drawLabel("4) See flower level and value rise", at: CGPoint(x: x + 14, y: y + h - 122), size: 11, color: .lightGray, weight: .regular)
    }

    func drawLabel(_ text: String, at point: CGPoint, size: CGFloat, color: NSColor, weight: NSFont.Weight) {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: size, weight: weight),
            .foregroundColor: color,
        ]
        NSAttributedString(string: text, attributes: attrs).draw(at: point)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: OverlayWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = OverlayWindow()
        print("[overlay] Widget overlay running")
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
