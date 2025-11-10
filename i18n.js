const I18N = (() => {
    const STORAGE_KEY = 'preferredLanguage';
    const DEFAULT_LANG = 'en';
    const SUPPORTED_LANGS = ['en', 'vi'];

    const translations = {
        en: {
            common: {
                language: 'Language',
                english: 'English',
                vietnamese: 'Vietnamese',
                backToDashboard: 'Back to Dashboard',
                settings: 'Settings',
                admin: 'Admin',
                selectLanguage: 'Select language',
                all: 'All',
                reload: 'Reload Stream',
                viewAll: 'View All',
                previous: 'Previous',
                next: 'Next',
                loading: 'Loading...',
                itemsPerPage: 'Items per page',
                reset: 'Reset',
                saveChanges: 'Save Changes',
                snapshotUnavailable: 'Snapshot unavailable',
                requestFailed: 'Request failed'
            },
            months: [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ],
            settings: {
                pageTitle: 'Monitor Settings',
                heading: 'Monitor Settings',
                description: 'Update configuration values stored in <code>config/config.yaml</code>. Changes save to disk and apply the next time the monitor restarts.',
                modelPath: 'Model Path',
                cameraIndex: 'Camera Index',
                imgSize: 'Image Size',
                confidenceThreshold: 'Confidence Threshold',
                redRatioThreshold: 'Red Ratio Threshold',
                cooldownSeconds: 'Cooldown Seconds',
                cooldownExpirySeconds: 'Cooldown Expiry Seconds',
                trackerDistance: 'Tracker Distance',
                schoolStartHour: 'School Start Hour',
                schoolEndHour: 'School End Hour',
                alarmFile: 'Alarm File',
                snapshotDir: 'Snapshot Directory',
                dbPath: 'Database Path',
                logCsvPath: 'Log CSV Path',
                studentSamplesDir: 'Student Samples Directory',
                attendanceConfidenceThreshold: 'Attendance Confidence Threshold',
                attendanceCooldownSeconds: 'Attendance Cooldown Seconds',
                recognitionMinSamples: 'Recognition Min Samples',
                faceDistanceThreshold: 'Face Distance Threshold',
                attendanceEnabled: 'Attendance Enabled',
                timeWindowEnabled: 'Time Window Enabled',
                soundAlert: 'Sound Alert',
                voiceAlert: 'Voice Alert',
                saveViolationImages: 'Save Violation Images',
                resetButton: 'Reset',
                saveButton: 'Save Changes',
                status: {
                    loading: 'Loading settings...',
                    loaded: 'Settings loaded.',
                    failed: 'Failed to load settings.',
                    saving: 'Saving...',
                    saveFailed: 'Unable to save settings.',
                    saveSuccess: 'Settings saved. Restart the monitor to apply changes.'
                },
                error: {
                    unableToFetch: 'Unable to fetch settings',
                    saveFailed: 'Save failed'
                }
            },
            webcam: {
                pageTitle: 'Student Monitor',
                heading: 'Student Monitor',
                stream: {
                    idle: 'Stream idle.',
                    connecting: 'Connecting to stream...',
                    online: 'Live stream active.',
                    offline: 'Stream disconnected. Retrying...'
                },
                stats: {
                    total: {
                        label: 'Total Persons',
                        hint: 'Detected in current frame'
                    },
                    withScarf: {
                        label: 'With Scarf',
                        hint: 'Compliant students'
                    },
                    missingScarf: {
                        label: 'Missing Scarf',
                        hint: 'Active violations'
                    },
                    recognized: {
                        label: 'Recognized',
                        hint: 'Identified students'
                    },
                    fps: {
                        label: 'Frame Rate',
                        hint: 'Frames per second'
                    },
                    status: {
                        label: 'Status',
                        initializing: 'Initializing...',
                        monitoring: 'Monitoring...',
                        idle: 'Idle',
                        unavailable: 'Status unavailable'
                    }
                },
                captured: {
                    heading: 'Captured Images',
                    badge: '{{count}} items',
                    empty: 'No captured images for this selection yet.',
                    error: 'Unable to load captured images right now.'
                },
                filters: {
                    year: 'Year',
                    month: 'Month',
                    day: 'Day'
                },
                captureToggle: {
                    start: 'Start',
                    stop: 'Stop',
                    error: 'Unable to update capture setting.'
                },
                attendanceFeed: {
                    heading: 'Recent Check-ins/Check-outs',
                    columns: {
                        time: 'Time',
                        student: 'Student',
                        event: 'Event'
                    },
                    empty: 'No attendance activity yet.',
                    disabled: 'Attendance tracking is disabled.',
                    error: 'Unable to load attendance activity right now.',
                    updated: 'Updated {{time}}',
                    unknownStudent: 'Unknown student',
                    eventType: {
                        checkIn: 'Check In',
                        checkOut: 'Check Out',
                        unknown: 'Unknown'
                    }
                },
                violation: {
                    alt: 'Violation at {{time}}',
                    tagDefault: 'NO_SCARF',
                    noSnapshot: 'Snapshot unavailable'
                }
            },
            captures: {
                pageTitle: 'Captured Violations',
                heading: 'Captured Violations',
                filters: {
                    year: 'Year',
                    month: 'Month',
                    day: 'Day',
                    startDate: 'Start Date',
                    endDate: 'End Date'
                },
                perPage: 'Items per page',
                summary: {
                    loading: 'Loading...',
                    text: '<strong>{{count}}</strong> captures — {{scope}}',
                    scope: {
                        allTime: 'All time',
                        year: 'Year {{year}}',
                        month: '{{month}}',
                        day: 'Day {{day}}',
                        from: 'from {{date}}',
                        to: 'to {{date}}'
                    }
                },
                grid: {
                    empty: 'No captures match the current filters yet.'
                },
                alert: {
                    invalidRange: 'Start date must be before end date.'
                },
                error: {
                    load: 'Unable to load captures right now.',
                    resetFilters: 'Reset filters'
                },
                pagination: {
                    info: 'Page {{page}} of {{totalPages}} | {{total}} total'
                }
            }
        },
        vi: {
            common: {
                language: 'Ngôn ngữ',
                english: 'Tiếng Anh',
                vietnamese: 'Tiếng Việt',
                backToDashboard: 'Quay lại Bảng điều khiển',
                settings: 'Cài đặt',
                admin: 'Quản trị',
                selectLanguage: 'Chọn ngôn ngữ',
                all: 'Tất cả',
                reload: 'Tải lại luồng',
                viewAll: 'Xem tất cả',
                previous: 'Trước',
                next: 'Tiếp',
                loading: 'Đang tải...',
                itemsPerPage: 'Số mục mỗi trang',
                reset: 'Đặt lại',
                saveChanges: 'Lưu thay đổi',
                snapshotUnavailable: 'Không có ảnh chụp',
                requestFailed: 'Yêu cầu thất bại'
            },
            months: [
                'Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12'
            ],
            settings: {
                pageTitle: 'Cài đặt Giám sát',
                heading: 'Cài đặt Giám sát',
                description: 'Cập nhật các giá trị cấu hình trong <code>config/config.yaml</code>. Thay đổi sẽ được lưu và áp dụng khi khởi động lại hệ thống.',
                modelPath: 'Đường dẫn mô hình',
                cameraIndex: 'Chỉ số camera',
                imgSize: 'Kích thước ảnh',
                confidenceThreshold: 'Ngưỡng độ tin cậy',
                redRatioThreshold: 'Ngưỡng tỉ lệ màu đỏ',
                cooldownSeconds: 'Thời gian chờ (giây)',
                cooldownExpirySeconds: 'Thời gian hết hạn chờ (giây)',
                trackerDistance: 'Khoảng cách theo dõi',
                schoolStartHour: 'Giờ bắt đầu học',
                schoolEndHour: 'Giờ kết thúc học',
                alarmFile: 'Tệp âm thanh cảnh báo',
                snapshotDir: 'Thư mục lưu ảnh',
                dbPath: 'Đường dẫn cơ sở dữ liệu',
                logCsvPath: 'Đường dẫn tệp CSV nhật ký',
                studentSamplesDir: 'Thư mục mẫu học sinh',
                attendanceConfidenceThreshold: 'Ngưỡng tin cậy điểm danh',
                attendanceCooldownSeconds: 'Thời gian chờ điểm danh (giây)',
                recognitionMinSamples: 'Số mẫu tối thiểu nhận diện',
                faceDistanceThreshold: 'Ngưỡng khoảng cách khuôn mặt',
                attendanceEnabled: 'Bật điểm danh',
                timeWindowEnabled: 'Bật khung thời gian',
                soundAlert: 'Cảnh báo âm thanh',
                voiceAlert: 'Cảnh báo giọng nói',
                saveViolationImages: 'Lưu ảnh vi phạm',
                resetButton: 'Đặt lại',
                saveButton: 'Lưu thay đổi',
                status: {
                    loading: 'Đang tải cài đặt...',
                    loaded: 'Đã tải cài đặt.',
                    failed: 'Không tải được cài đặt.',
                    saving: 'Đang lưu...',
                    saveFailed: 'Không thể lưu cài đặt.',
                    saveSuccess: 'Đã lưu cài đặt. Vui lòng khởi động lại hệ thống để áp dụng.'
                },
                error: {
                    unableToFetch: 'Không thể lấy cài đặt',
                    saveFailed: 'Lưu thất bại'
                }
            },
            webcam: {
                pageTitle: 'Giám sát học sinh',
                heading: 'Giám sát học sinh',
                stream: {
                    idle: 'Luồng đang chờ.',
                    connecting: 'Đang kết nối tới luồng...',
                    online: 'Luồng trực tiếp đang hoạt động.',
                    offline: 'Luồng bị ngắt. Đang thử lại...'
                },
                stats: {
                    total: {
                        label: 'Tổng số người',
                        hint: 'Được phát hiện trong khung hình hiện tại'
                    },
                    withScarf: {
                        label: 'Đeo khăn',
                        hint: 'Học sinh tuân thủ'
                    },
                    missingScarf: {
                        label: 'Thiếu khăn',
                        hint: 'Vi phạm đang xảy ra'
                    },
                    recognized: {
                        label: 'Đã nhận diện',
                        hint: 'Học sinh đã nhận diện'
                    },
                    fps: {
                        label: 'Tốc độ khung hình',
                        hint: 'Số khung hình mỗi giây'
                    },
                    status: {
                        label: 'Trạng thái',
                        initializing: 'Đang khởi tạo...',
                        monitoring: 'Đang giám sát...',
                        idle: 'Nghỉ',
                        unavailable: 'Không có thông tin trạng thái'
                    }
                },
                captured: {
                    heading: 'Ảnh đã chụp',
                    badge: '{{count}} mục',
                    empty: 'Chưa có ảnh nào cho lựa chọn này.',
                    error: 'Hiện không thể tải ảnh đã chụp.'
                },
                filters: {
                    year: 'Năm',
                    month: 'Tháng',
                    day: 'Ngày'
                },
                captureToggle: {
                    start: 'Bật',
                    stop: 'Tắt',
                    error: 'Không thể cập nhật trạng thái chụp.'
                },
                attendanceFeed: {
                    heading: 'Điểm danh gần đây',
                    columns: {
                        time: 'Thời gian',
                        student: 'Học sinh',
                        event: 'Sự kiện'
                    },
                    empty: 'Chưa có hoạt động điểm danh.',
                    disabled: 'Chức năng điểm danh đang tắt.',
                    error: 'Không thể tải hoạt động điểm danh.',
                    updated: 'Cập nhật {{time}}',
                    unknownStudent: 'Học sinh không xác định',
                    eventType: {
                        checkIn: 'Điểm danh vào',
                        checkOut: 'Điểm danh ra',
                        unknown: 'Không xác định'
                    }
                },
                violation: {
                    alt: 'Vi phạm lúc {{time}}',
                    tagDefault: 'KHÔNG_KHAN',
                    noSnapshot: 'Không có ảnh chụp'
                }
            },
            captures: {
                pageTitle: 'Danh sách Vi phạm',
                heading: 'Danh sách Vi phạm',
                filters: {
                    year: 'Năm',
                    month: 'Tháng',
                    day: 'Ngày',
                    startDate: 'Ngày bắt đầu',
                    endDate: 'Ngày kết thúc'
                },
                perPage: 'Số mục mỗi trang',
                summary: {
                    loading: 'Đang tải...',
                    text: '<strong>{{count}}</strong> ảnh vi phạm — {{scope}}',
                    scope: {
                        allTime: 'Toàn bộ thời gian',
                        year: 'Năm {{year}}',
                        month: '{{month}}',
                        day: 'Ngày {{day}}',
                        from: 'từ {{date}}',
                        to: 'đến {{date}}'
                    }
                },
                grid: {
                    empty: 'Không có ảnh nào khớp với bộ lọc hiện tại.'
                },
                alert: {
                    invalidRange: 'Ngày bắt đầu phải trước ngày kết thúc.'
                },
                error: {
                    load: 'Hiện không thể tải danh sách vi phạm.',
                    resetFilters: 'Đặt lại bộ lọc'
                },
                pagination: {
                    info: 'Trang {{page}} / {{totalPages}} | Tổng {{total}}'
                }
            }
        }
    };

    const subscriptions = new Set();
    let currentLanguage = DEFAULT_LANG;

    function getStoredLanguage() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored && SUPPORTED_LANGS.includes(stored)) {
                return stored;
            }
        } catch (err) {
            // ignore storage errors
        }
        return DEFAULT_LANG;
    }

    function storeLanguage(lang) {
        try {
            localStorage.setItem(STORAGE_KEY, lang);
        } catch (err) {
            // ignore storage errors
        }
    }

    function resolveKey(key, lang) {
        const parts = key.split('.');
        let value = translations[lang];
        for (const part of parts) {
            if (value == null) {
                return undefined;
            }
            value = value[part];
        }
        if (value === undefined && lang !== DEFAULT_LANG) {
            return resolveKey(key, DEFAULT_LANG);
        }
        return value;
    }

    function applyReplacements(text, replacements) {
        if (!replacements) return text;
        return Object.keys(replacements).reduce((result, key) => {
            const value = replacements[key];
            return result.replace(new RegExp(`{{\\s*${key}\\s*}}`, 'g'), value);
        }, text);
    }

    function getTranslation(key, replacements, lang = currentLanguage) {
        const value = resolveKey(key, lang);
        if (typeof value === 'string') {
            return applyReplacements(value, replacements);
        }
        return value;
    }

    function formatNumber(value, options) {
        const locale = currentLanguage === 'vi' ? 'vi-VN' : 'en-US';
        const formatter = new Intl.NumberFormat(locale, options);
        return formatter.format(value ?? 0);
    }

    function formatDate(date, options) {
        const locale = currentLanguage === 'vi' ? 'vi-VN' : 'en-US';
        return date.toLocaleString(locale, options);
    }

    function getMonthName(monthNumber, lang = currentLanguage) {
        const monthIndex = Number(monthNumber) - 1;
        const monthNames = resolveKey('months', lang);
        if (Array.isArray(monthNames) && monthNames[monthIndex]) {
            return monthNames[monthIndex];
        }
        if (Array.isArray(monthNames) && monthNames[monthIndex] === undefined && lang !== DEFAULT_LANG) {
            return getMonthName(monthNumber, DEFAULT_LANG);
        }
        return monthNumber;
    }

    function parseArgs(value) {
        if (!value) return undefined;
        try {
            return JSON.parse(value);
        } catch (err) {
            return undefined;
        }
    }

    function translateElement(el) {
        const textKey = el.dataset.i18n;
        if (textKey) {
            const args = parseArgs(el.dataset.i18nArgs);
            const translated = getTranslation(textKey, args);
            if (typeof translated === 'string') {
                el.textContent = translated;
            }
        }

        const htmlKey = el.dataset.i18nHtml;
        if (htmlKey) {
            const args = parseArgs(el.dataset.i18nHtmlArgs);
            const translated = getTranslation(htmlKey, args);
            if (typeof translated === 'string') {
                el.innerHTML = translated;
            }
        }

        const placeholderKey = el.dataset.i18nPlaceholder;
        if (placeholderKey) {
            const args = parseArgs(el.dataset.i18nPlaceholderArgs);
            const translated = getTranslation(placeholderKey, args);
            if (typeof translated === 'string') {
                el.setAttribute('placeholder', translated);
            }
        }

        const valueKey = el.dataset.i18nValue;
        if (valueKey) {
            const args = parseArgs(el.dataset.i18nValueArgs);
            const translated = getTranslation(valueKey, args);
            if (typeof translated === 'string') {
                el.value = translated;
            }
        }

        const titleKey = el.dataset.i18nTitle;
        if (titleKey) {
            const args = parseArgs(el.dataset.i18nTitleArgs);
            const translated = getTranslation(titleKey, args);
            if (typeof translated === 'string') {
                el.setAttribute('title', translated);
            }
        }

        const ariaLabelKey = el.dataset.i18nAriaLabel;
        if (ariaLabelKey) {
            const args = parseArgs(el.dataset.i18nAriaLabelArgs);
            const translated = getTranslation(ariaLabelKey, args);
            if (typeof translated === 'string') {
                el.setAttribute('aria-label', translated);
            }
        }

        const altKey = el.dataset.i18nAlt;
        if (altKey) {
            const args = parseArgs(el.dataset.i18nAltArgs);
            const translated = getTranslation(altKey, args);
            if (typeof translated === 'string') {
                el.setAttribute('alt', translated);
            }
        }
    }

    function translatePage() {
        document.documentElement.setAttribute('lang', currentLanguage);
        const titleEl = document.querySelector('title[data-i18n]');
        if (titleEl) {
            const args = parseArgs(titleEl.dataset.i18nArgs);
            const translated = getTranslation(titleEl.dataset.i18n, args);
            if (typeof translated === 'string') {
                titleEl.textContent = translated;
            }
        }

        const elements = document.querySelectorAll('[data-i18n], [data-i18n-html], [data-i18n-placeholder], [data-i18n-value], [data-i18n-title], [data-i18n-aria-label], [data-i18n-alt]');
        elements.forEach(translateElement);

        updateLanguageSelect();
    }

    function updateLanguageSelect() {
        const select = document.getElementById('languageSelect');
        if (!select) return;

        select.setAttribute('aria-label', getTranslation('common.selectLanguage'));

        select.querySelectorAll('option').forEach((option) => {
            const value = option.value;
            if (value === 'en') {
                option.textContent = getTranslation('common.english');
            } else if (value === 'vi') {
                option.textContent = getTranslation('common.vietnamese');
            }
        });

        if (select.value !== currentLanguage) {
            select.value = currentLanguage;
        }
    }

    function setLanguage(lang) {
        if (!SUPPORTED_LANGS.includes(lang)) {
            lang = DEFAULT_LANG;
        }
        if (lang === currentLanguage) {
            translatePage();
            return;
        }
        currentLanguage = lang;
        storeLanguage(lang);
        translatePage();
        subscriptions.forEach((callback) => {
            try {
                callback(lang);
            } catch (err) {
                // ignore subscriber errors
            }
        });
    }

    function getLanguage() {
        return currentLanguage;
    }

    function onChange(callback) {
        if (typeof callback === 'function') {
            subscriptions.add(callback);
            return () => subscriptions.delete(callback);
        }
        return () => {};
    }

    function initLanguageSelect() {
        const select = document.getElementById('languageSelect');
        if (!select) return;
        updateLanguageSelect();
        select.addEventListener('change', (event) => {
            const selected = event.target.value;
            setLanguage(selected);
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        currentLanguage = getStoredLanguage();
        translatePage();
        initLanguageSelect();
    });

    return {
        t: getTranslation,
        get: getTranslation,
        resolve: (key) => resolveKey(key, currentLanguage),
        setLanguage,
        getLanguage,
        onChange,
        formatNumber,
        formatDate,
        getMonthName
    };
})();

window.I18N = I18N;
window.t = I18N.t;

