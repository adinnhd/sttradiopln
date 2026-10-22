// Tampilkan satu kelompok driver Windows agar perangkat tidak berulang.
// ID asli tetap dipakai saat membuka stream; daftar lengkap selalu tersedia.
export function visibleDevices(devices, showAll = false) {
  if (showAll) return devices;
  for (const driver of [/wasapi/i, /directsound/i, /^mme$/i]) {
    const group = devices.filter(device => driver.test(device.hostapi));
    if (group.length) {
      const inputs = group.filter(device => !/^(Microsoft Sound Mapper - Input|Primary Sound Capture Driver)$/i.test(device.name));
      return inputs.length ? inputs : group;
    }
  }
  return devices;
}
