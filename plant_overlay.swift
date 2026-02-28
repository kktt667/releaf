import Cocoa

enum Stage: String {
    case normal
    case warning
    case decay
    case recovery
}

struct OverlayStatus {
    var state: Stage = .normal
    var chainBadge: String = "CHAIN_FALLBACK_LOCAL_LEDGER"
    var faceDetected: Bool = false
    var eyesDetected: Bool = false
    var lookingScore: Double = 0.0
    var screenTimeSeconds: Double = 0.0
    var focusStreakSeconds: Double = 0.0
    var statusMessage: String = ""
    var demoStep: String = ""
    var flowerLevel: Int = 1
    var healthScore: Int = 56
    var valueScore: Double = 0.0
    var sessions: Int = 0
    var streakDays: Int = 0
    var decayPenalties: Int = 0
    var tokenId: String = ""
    var txHash: String = ""
    var proofURL: String = ""
    var localBlockHash: String = ""
    var mintFlashUntil: Double = 0.0
    var qrFile: String = "data/current_qr.png"
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
    var pulse: CGFloat = 0.0
    var status = OverlayStatus()
    var qrImage: NSImage?
    var lastQrPath: String = ""

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.readStatus()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func tick() {
        pulse += 0.08
        if pulse > 10_000 { pulse = 0 }
        needsDisplay = true
    }

    func readStatus() {
        let path = FileManager.default.currentDirectoryPath + "/data/overlay_status.json"
        guard
            let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        if let stateRaw = json["state"] as? String, let st = Stage(rawValue: stateRaw.lowercased()) {
            status.state = st
        }
        status.chainBadge = json["chain_badge"] as? String ?? status.chainBadge
        status.faceDetected = json["face_detected"] as? Bool ?? false
        status.eyesDetected = json["eyes_detected"] as? Bool ?? false
        status.lookingScore = json["looking_score"] as? Double ?? 0.0
        status.screenTimeSeconds = json["screen_time_seconds"] as? Double ?? 0.0
        status.focusStreakSeconds = json["focus_streak_seconds"] as? Double ?? 0.0
        status.statusMessage = json["status_message"] as? String ?? ""
        status.demoStep = json["demo_step"] as? String ?? ""
        status.flowerLevel = json["flower_level"] as? Int ?? 1
        status.healthScore = json["health_score"] as? Int ?? 56
        status.valueScore = json["value_score"] as? Double ?? 0.0
        status.sessions = json["sessions"] as? Int ?? 0
        status.streakDays = json["streak_days"] as? Int ?? 0
        status.decayPenalties = json["decay_penalties"] as? Int ?? 0
        status.tokenId = json["token_id"] as? String ?? ""
        status.txHash = json["tx_hash"] as? String ?? ""
        status.proofURL = json["proof_url"] as? String ?? ""
        status.localBlockHash = json["local_block_hash"] as? String ?? ""
        status.mintFlashUntil = json["mint_flash_until"] as? Double ?? 0.0
        status.qrFile = json["qr_file"] as? String ?? status.qrFile

        let qrPath = status.qrFile.hasPrefix("/") ? status.qrFile : (FileManager.default.currentDirectoryPath + "/" + status.qrFile)
        if qrPath != lastQrPath || qrImage == nil {
            qrImage = NSImage(contentsOfFile: qrPath)
            lastQrPath = qrPath
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width
        let h = bounds.height

        drawStageEffects(ctx: ctx, w: w, h: h)

        // Left stacked widgets; camera app window sits below these.
        drawStatusCard(ctx: ctx, rect: CGRect(x: 14, y: h - 162, width: 320, height: 146))
        drawStatsCard(ctx: ctx, rect: CGRect(x: 14, y: h - 330, width: 320, height: 154))

        // Right widgets with QR + chain.
        drawQRCard(ctx: ctx, rect: CGRect(x: w - 282, y: h - 302, width: 268, height: 286))
        drawChainCard(ctx: ctx, rect: CGRect(x: w - 282, y: h - 472, width: 268, height: 156))

        drawMintPopup(ctx: ctx, w: w, h: h)
    }

    func drawStageEffects(ctx: CGContext, w: CGFloat, h: CGFloat) {
        let phase = (sin(pulse) + 1.0) * 0.5
        switch status.state {
        case .normal:
            ctx.setFillColor(NSColor(calibratedWhite: 0.0, alpha: 0.04).cgColor)
            ctx.fill(bounds)
        case .warning:
            ctx.setFillColor(NSColor(calibratedRed: 0.95, green: 0.75, blue: 0.25, alpha: 0.15 + (0.08 * phase)).cgColor)
            ctx.fill(bounds)
        case .decay:
            ctx.setFillColor(NSColor(calibratedRed: 0.75, green: 0.35, blue: 0.40, alpha: 0.20 + (0.10 * phase)).cgColor)
            ctx.fill(bounds)
            ctx.setStrokeColor(NSColor(calibratedRed: 0.88, green: 0.50, blue: 0.58, alpha: 0.55).cgColor)
            ctx.setLineWidth(10)
            ctx.stroke(bounds.insetBy(dx: 8, dy: 8))
        case .recovery:
            ctx.setFillColor(NSColor(calibratedRed: 0.24, green: 0.64, blue: 0.96, alpha: 0.15 + (0.10 * phase)).cgColor)
            ctx.fill(bounds)
            let stripeX = ((w + 200) * phase) - 180
            ctx.setFillColor(NSColor(calibratedRed: 0.75, green: 1.0, blue: 0.95, alpha: 0.16).cgColor)
            ctx.fill(CGRect(x: stripeX, y: 0, width: 180, height: h))
        }
    }

    func drawCard(ctx: CGContext, rect: CGRect, title: String) {
        let path = CGPath(roundedRect: rect, cornerWidth: 16, cornerHeight: 16, transform: nil)
        ctx.addPath(path)
        ctx.setFillColor(NSColor(calibratedRed: 0.10, green: 0.12, blue: 0.19, alpha: 0.82).cgColor)
        ctx.fillPath()
        ctx.addPath(path)
        ctx.setStrokeColor(NSColor(calibratedRed: 0.50, green: 0.56, blue: 0.72, alpha: 0.70).cgColor)
        ctx.setLineWidth(1.8)
        ctx.strokePath()
        drawLabel(title, at: CGPoint(x: rect.minX + 12, y: rect.maxY - 30), size: 15, color: .white, weight: .semibold)
    }

    func drawStatusCard(ctx: CGContext, rect: CGRect) {
        drawCard(ctx: ctx, rect: rect, title: "Garden State")
        drawLabel(status.state.rawValue.uppercased(), at: CGPoint(x: rect.minX + 14, y: rect.maxY - 64), size: 22, color: stageColor(), weight: .bold)
        let eyeColor = status.eyesDetected ? NSColor.systemGreen : NSColor.systemRed
        ctx.setFillColor(eyeColor.cgColor)
        ctx.fillEllipse(in: CGRect(x: rect.minX + 16, y: rect.minY + 30, width: 13, height: 13))
        drawLabel("Eyes \(status.eyesDetected ? "Seen" : "Searching")", at: CGPoint(x: rect.minX + 35, y: rect.minY + 28), size: 12, color: .lightGray, weight: .regular)
        drawLabel("Chain: \(status.chainBadge)", at: CGPoint(x: rect.minX + 14, y: rect.minY + 10), size: 11, color: .lightGray, weight: .regular)
    }

    func drawStatsCard(ctx: CGContext, rect: CGRect) {
        drawCard(ctx: ctx, rect: rect, title: "Live Stats")
        drawLabel(String(format: "Focus %.2f   Time %.1fm", status.lookingScore, status.screenTimeSeconds / 60.0),
                  at: CGPoint(x: rect.minX + 14, y: rect.maxY - 58),
                  size: 12,
                  color: .lightGray,
                  weight: .regular)
        drawLabel(String(format: "HP %d%%   Value $%.2f", status.healthScore, status.valueScore * 10.0),
                  at: CGPoint(x: rect.minX + 14, y: rect.maxY - 82),
                  size: 12,
                  color: .lightGray,
                  weight: .regular)
        drawLabel("Level \(status.flowerLevel)  Sessions \(status.sessions)", at: CGPoint(x: rect.minX + 14, y: rect.maxY - 106), size: 12, color: .lightGray, weight: .regular)
        drawLabel("Streak \(status.streakDays)d  Decay \(status.decayPenalties)", at: CGPoint(x: rect.minX + 14, y: rect.maxY - 128), size: 12, color: .lightGray, weight: .regular)
        let line = status.statusMessage.isEmpty ? status.demoStep : status.statusMessage
        drawLabel(line, at: CGPoint(x: rect.minX + 14, y: rect.minY + 10), size: 11, color: .white, weight: .regular)
    }

    func drawQRCard(ctx: CGContext, rect: CGRect) {
        drawCard(ctx: ctx, rect: rect, title: "Photo Check-in QR")
        if let qr = qrImage {
            let imageRect = CGRect(x: rect.minX + 30, y: rect.minY + 72, width: rect.width - 60, height: rect.width - 60)
            qr.draw(in: imageRect)
            ctx.setStrokeColor(NSColor(calibratedRed: 0.72, green: 0.77, blue: 0.95, alpha: 0.95).cgColor)
            ctx.stroke(imageRect.insetBy(dx: -2, dy: -2))
        } else {
            drawLabel("Waiting for QR...", at: CGPoint(x: rect.minX + 26, y: rect.minY + 120), size: 13, color: .lightGray, weight: .regular)
        }
        drawLabel("Press Y -> scan -> upload", at: CGPoint(x: rect.minX + 18, y: rect.minY + 42), size: 12, color: .lightGray, weight: .regular)
        drawLabel("Then watch NFT popup", at: CGPoint(x: rect.minX + 18, y: rect.minY + 22), size: 12, color: .lightGray, weight: .regular)
    }

    func drawChainCard(ctx: CGContext, rect: CGRect) {
        drawCard(ctx: ctx, rect: rect, title: "NFT + Immutable Chain")
        drawLabel("Token: \(status.tokenId.isEmpty ? "pending" : status.tokenId)", at: CGPoint(x: rect.minX + 14, y: rect.maxY - 56), size: 11, color: .lightGray, weight: .regular)
        drawLabel("Tx: \(status.txHash.prefix(18))", at: CGPoint(x: rect.minX + 14, y: rect.maxY - 78), size: 11, color: .lightGray, weight: .regular)
        drawLabel("Block: \(status.localBlockHash.prefix(24))", at: CGPoint(x: rect.minX + 14, y: rect.maxY - 100), size: 11, color: .lightGray, weight: .regular)
        if !status.proofURL.isEmpty {
            drawLabel("Proof URL ready in /my-flower", at: CGPoint(x: rect.minX + 14, y: rect.minY + 10), size: 11, color: .white, weight: .regular)
        }
    }

    func drawMintPopup(ctx: CGContext, w: CGFloat, h: CGFloat) {
        let now = Date().timeIntervalSince1970
        guard status.mintFlashUntil > now else { return }
        let phase = CGFloat((status.mintFlashUntil - now) / 6.0)
        let alpha = max(0.3, min(1.0, phase))
        let rect = CGRect(x: max(40, (w / 2) - 250), y: max(90, (h / 2) - 90), width: 500, height: 180)
        let path = CGPath(roundedRect: rect, cornerWidth: 18, cornerHeight: 18, transform: nil)
        ctx.addPath(path)
        ctx.setFillColor(NSColor(calibratedRed: 0.12, green: 0.45, blue: 0.24, alpha: alpha).cgColor)
        ctx.fillPath()
        ctx.addPath(path)
        ctx.setStrokeColor(NSColor(calibratedRed: 0.70, green: 0.95, blue: 0.78, alpha: alpha).cgColor)
        ctx.setLineWidth(3)
        ctx.strokePath()
        drawLabel("NFT UPGRADED", at: CGPoint(x: rect.minX + 124, y: rect.maxY - 64), size: 33, color: .white, weight: .bold)
        drawLabel(String(format: "Value $%.2f  |  Level %d", status.valueScore * 10.0, status.flowerLevel), at: CGPoint(x: rect.minX + 126, y: rect.maxY - 98), size: 16, color: .white, weight: .semibold)
        drawLabel("Open /my-flower on phone", at: CGPoint(x: rect.minX + 130, y: rect.maxY - 124), size: 14, color: .white, weight: .regular)
    }

    func stageColor() -> NSColor {
        switch status.state {
        case .normal: return NSColor(calibratedRed: 0.42, green: 0.80, blue: 0.52, alpha: 0.95)
        case .warning: return NSColor(calibratedRed: 0.95, green: 0.74, blue: 0.28, alpha: 0.95)
        case .decay: return NSColor(calibratedRed: 0.92, green: 0.46, blue: 0.52, alpha: 0.95)
        case .recovery: return NSColor(calibratedRed: 0.38, green: 0.72, blue: 0.96, alpha: 0.95)
        }
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
        print("[overlay] Widget overlay running with QR/status sync")
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
