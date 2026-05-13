# Turn off motherboard + GPU RGB

Hardware:
- Motherboard: ASRock B650M Pro RS WiFi (Polychrome USB controller)
- GPU: Gigabyte GeForce RTX 3060 Gaming OC V2 (RGB Fusion, I²C)

Tool: OpenRGB 1.0rc2 AppImage, installed at `/opt/openrgb/OpenRGB.AppImage` and symlinked to `/usr/local/bin/openrgb`.

## One-time setup (per boot, if not made persistent)

```bash
sudo modprobe i2c-dev
```

To make the module load automatically on every boot:

```bash
echo 'i2c-dev' | sudo tee /etc/modules-load.d/i2c-dev.conf
```

## Turn off the lights

```bash
sudo openrgb --device 1 --mode Off
sudo openrgb --device 0 --mode Direct --color 000000
```

- Device 1 = motherboard (has a native `Off` mode)
- Device 0 = GPU (no `Off` mode; using `Direct` with black color is the equivalent)

Device numbers come from `sudo openrgb --list-devices` and could change if hardware is added/removed.

## Persistent: run automatically at boot and after resume

A systemd unit at `/etc/systemd/system/openrgb-off.service` runs the two commands at boot and again whenever the system resumes from suspend / hibernate:

```ini
[Unit]
Description=Turn off motherboard and GPU RGB via OpenRGB
After=multi-user.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

[Service]
Type=oneshot
ExecStartPre=-/sbin/modprobe i2c-dev
ExecStart=/usr/local/bin/openrgb --device 1 --mode Off
ExecStart=/usr/local/bin/openrgb --device 0 --mode Direct --color 000000

[Install]
WantedBy=multi-user.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
```

Install / enable:

```bash
sudo install -m 0644 /tmp/openrgb-off.service /etc/systemd/system/openrgb-off.service
sudo systemctl daemon-reload
sudo systemctl enable --now openrgb-off.service
```

Check it works:

```bash
systemctl status openrgb-off.service
journalctl -u openrgb-off.service -n 20
```

If device numbers ever change (hardware added/removed), update the `ExecStart=` lines after re-checking `sudo openrgb --list-devices`, then `sudo systemctl daemon-reload && sudo systemctl restart openrgb-off.service`.

## Notes

- These settings live in volatile controller memory. After a full reboot, sleep/wake, or BIOS POST they may return; just re-run the two commands.
- The `[i2c_smbus_linux] Failed to read i2c device PCI device ID` warnings during probing are harmless.
- The "udev rules not installed" warning only matters if you want to run OpenRGB without `sudo`. Not needed for the off-commands above.
- For a permanent motherboard-side fix, ASRock UEFI has a Polychrome RGB / Onboard LED toggle that disables board LEDs at the firmware level (does not affect the GPU).
